# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import *
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from typing import Union
import logging
import torch
from transformers import BertModel, BertConfig, BertTokenizer
from customized_ctc import ctc_loss as custom_ctc_loss
from tqdm import tqdm
import numpy as np
import itertools
import time
import nltk
from nltk.util import ngrams

logger = logging.getLogger(__name__)


def unravel_indices(indices: torch.LongTensor,
        shape: Tuple[int, ...],
) -> torch.LongTensor:
    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
                (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = utils.new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


def get_n_gram_counts(source, summary, n):
    count = 0
    source_n_gram = list(ngrams(source, n))
    summary_n_gram = list(ngrams(summary, n))
    for gram in summary_n_gram:
        if gram in source_n_gram:
            count += 1
    return count

@register_model("nat_encoder_only_customized_ctc")
class NATransformerModel(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.bert_tokenizer = BertTokenizer.from_pretrained('my_bert_tokenizer')
        self.bert_decode_function = None
        if args.plain_ctc == False and (args.length_control == "truncate" or args.length_control == "no_control"):
            # we import the CTC beam decoder only if we don't use plain-ctc, truncate, or no length control
            from ctcdecode import CTCBeamDecoder
            self.ctc_decoder = CTCBeamDecoder(
                decoder.dictionary.symbols,
                model_path=None,
                alpha=0,
                beta=0,
                cutoff_top_n=40,
                cutoff_prob=1,
                beam_width=5000,
                num_processes=1,
                blank_id=decoder.dictionary.blank_index,
                log_probs_input=False
            )
        else:
            self.ctc_decoder=None
        self.copy_src_token = getattr(args, 'copy_src_token', False)
        self.plain_ctc = getattr(args, 'plain_ctc', False)
        self.custom_ctc = getattr(args, 'custom_ctc', False)
        self.length_control = getattr(args, 'length_control', None)
        self.desired_length = getattr(args, 'desired_length', None)
        self.marg_criteria = getattr(args, 'marg_criteria', None)
        self.scope = getattr(args, 'scope', None)
        self.k = getattr(args, 'k', None)
        self.beam_size = getattr(args, 'beam_size', None)
        self.use_bert_tokens = getattr(args, 'use_bert_tokens', False)
        self.use_bert_weights = False
        self.use_length_ratio = getattr(args, 'use_length_ratio', False)

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )
        parser.add_argument(
            "--src-upsample-scale",
            type=int,
            default=1
        )
        parser.add_argument(
            '--use-ctc-decoder',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--ctc-beam-size',
            default=1,
            type=int
        )
        parser.add_argument(
            '--ctc-beam-size-train',
            default=1,
            type=int
        )
        parser.add_argument(
            '--copy-src-token',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--softcopy',
            action='store_true',
            default=False
        )
        parser.add_argument(
            '--softcopy-temp',
            default=5,
            type=float
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def set_decode_function(self, decode_function):
        self.decode_function = decode_function

    def sequence_ctc_loss_with_logits(self,
                                      logits: torch.FloatTensor,
                                      logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      targets: torch.LongTensor,
                                      target_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      blank_index: torch.LongTensor,
                                      label_smoothing=0,
                                      reduce=True
                                      ) -> torch.FloatTensor:
        # lengths : (batch_size, )
        # calculated by counting number of mask
        logit_lengths = (logit_mask.bool()).long().sum(1)

        if len(targets.size()) == 1:
            targets = targets.unsqueeze(0)
            target_mask = target_mask.unsqueeze(0)
        target_lengths = (target_mask.bool()).long().sum(1)

        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        # log_probs_T : (T, batch_size, n_class), this kind of shape is required for ctc_loss
        log_probs_T = log_probs.transpose(0, 1)
        #     assert (target_lengths == 0).any()
        targets = targets.long()
        custom_targets = targets.clone()
        targets = targets[target_mask.bool()]
        if reduce:
            loss = custom_ctc_loss(log_probs_T.float(),  # compatible with fp16
                custom_targets,
                logit_lengths,
                target_lengths,
                blank=blank_index,
                reduction="mean",)
            loss = torch.stack([a / b for a, b in zip(loss, target_lengths)])
            loss = torch.clip(loss, min=0, max=100)  # Clip the loss to avoid overflow
            loss = loss.mean()
        else:
            loss = custom_ctc_loss(
                    log_probs_T.float(),  # compatible with fp16
                    custom_targets,
                    logit_lengths,
                    target_lengths,
                    blank=blank_index,
                    reduction="none",
                )
            loss = torch.clip(loss, min=0, max=100)  # Clip the loss to avoid overflow
            loss = torch.stack([a / b for a, b in zip(loss, target_lengths)])

        n_invalid_samples = (logit_lengths < target_lengths).long().sum()

        if n_invalid_samples > 0:
            logger.warning(
                f"The length of predicted alignment is shoter than target length, increase upsample factor: {n_invalid_samples} samples"
            )
            # raise ValueError

        if label_smoothing > 0:
            smoothed_loss = -log_probs.mean(-1)[logit_mask.bool()].mean()
            loss = (1 - label_smoothing) * loss + label_smoothing * smoothed_loss
        return loss

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, reduce=True, train_ratio=None, **kwargs
    ):

        # encoding
        prev_output_tokens = self.initialize_output_tokens_by_src_tokens(src_tokens)
        prev_output_tokens_mask = prev_output_tokens.ne(self.pad)
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

            # decoding
        word_ins_out = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
        )

        target_mask = tgt_tokens.ne(self.pad)
        # if self.args.use_ctc:
        ctc_loss = self.sequence_ctc_loss_with_logits(
            logits=word_ins_out,
            logit_mask=prev_output_tokens_mask,
            targets=tgt_tokens,
            target_mask=target_mask,
            blank_index=self.tgt_dict.blank_index,
            label_smoothing=self.args.label_smoothing,
            reduce=reduce
        )

        ret_val = {
            "ctc_loss": {"loss": ctc_loss},
        }
        if self.decoder.softcopy_learnable:
            ret_val.update(
                {
                    "stat:softcopy_temp": self.decoder.para_softcopy_temp.item()
                }
            )

        return ret_val

    def margin_over_prob_table(self, column, dp_log_prob_table, scope, k, path_num=-1, end_row=0, criteria="mean"):
        """
        This function marginalize over the first dimension of a k*k*...*k probability tensor based on the given criteria.
        The function returns the chosen token at the first dimension and a modified probability table such that all
        remaining dimensions except from the first dimension is shifted (e.g., the second dimension now becomes the first
        dimension). Furthermore, this probability table copies the the probability except the first axis and repeat it
        across the last dimension of the table (e.g., table[0, 0, 0] == table [0, 0, 1].... if the score is 3.

        prob_table_slot (torch.tensor of size [k]*scope):  the (log) probability of (top k) tokens at last (scope) time steps
        scope (int): dimensions of the prob distribution
        k (int): the size of the word distribution that we are tracking
        criteria (string): in which criteria do we preform the marginalization (currently we only support mean and max)

        returns:
        chosen_token_index (int): the chosen token at the first axis (which is the axis being marginalized)
        new_prob_table (torch.tensor of size [k]*scope): the new probability table after marginalization
        """
        # We take a mean of the probability of all sub-sequences starting from each token in the axis that
        # we are going to marginalize out
        remaining_axis = tuple(range(3, scope+2))  # A tuple of the remaining axis after marginalization
        if criteria == "mean":
            # If we are using mean as the marginalization selection criteria
            # we find the sum probability over all sequences starting from each token in the first axis
            # (since each sub-sequence has the same number of elements, we skip the "mean" process to reduce
            # the chance of suffering from underflow.
            # Notice taking exp could lead to underflow of probability.
            sum_old_prob_along_remaining_axis = torch.logsumexp(dp_log_prob_table[0:end_row+1, column-1], dim=remaining_axis)
            # We find the word with the max-sum probability
            most_probable_word_index = torch.argmax(sum_old_prob_along_remaining_axis, dim=2)
            corresponding_prob = torch.amax(sum_old_prob_along_remaining_axis, dim=2)
        elif criteria == "filtered_mean":
            sum_old_prob_along_remaining_axis = torch.logsumexp(dp_log_prob_table[0:end_row+1, column-1], dim=remaining_axis)
            # We find the word with the max-sum probability
            non_inf_sum = (dp_log_prob_table[0:end_row+1, column-1] != float("-inf")).long().sum(remaining_axis)
            sum_old_prob_along_remaining_axis = sum_old_prob_along_remaining_axis- non_inf_sum.log()
            sum_old_prob_along_remaining_axis = torch.nan_to_num(sum_old_prob_along_remaining_axis, float("-inf"))
            most_probable_word_index = torch.argmax(sum_old_prob_along_remaining_axis, dim=2)
            corresponding_prob = torch.amax(sum_old_prob_along_remaining_axis, dim=2)
        elif criteria == "max":
            # If we are using max as the select criteria, we select the token in the first axis that can lead to the
            # sub-sequence with the maximum probability
            max_old_prob_along_remaining_axis = torch.amax(dp_log_prob_table[0:end_row+1, column-1], dim=remaining_axis)
            most_probable_word_index = torch.argmax(max_old_prob_along_remaining_axis, dim=2)
            corresponding_prob = torch.amax(max_old_prob_along_remaining_axis, dim=2)
        elif criteria == "beam_search":
            max_old_prob_along_remaining_axis = dp_log_prob_table[0:end_row+1, column-1]
            most_probable_word_index = torch.argmax(max_old_prob_along_remaining_axis, dim=2)
            corresponding_prob = torch.amax(max_old_prob_along_remaining_axis, dim=2)
        else:
            most_probable_word_index = None
            raise NotImplementedError("Haven't designed other evaluation criteria")
        # We make a clone of the old probability such that we can modify this tensor
        # Notice we only select the sub-tensor starting from the selected token
        prob_size = dp_log_prob_table.size()
        new_prob_index = [torch.arange(0, end_row+1).repeat(prob_size[2], 1).transpose(0, 1).flatten(),
                          column - 1,
                          torch.arange(0, prob_size[2]).repeat(end_row+1, 1).flatten(),
                          most_probable_word_index.view(-1)]
        new_probability = dp_log_prob_table[new_prob_index].clone()
        new_probability = new_probability.reshape([end_row+1, path_num]+ list(prob_size[4:]))
        new_probability = new_probability.unsqueeze(-1)
        # the new dimension is in the wrong shape, we repeat it for k times to keep our data structure
        # we calculate the desired repeat-index to repeat for k times along the last dimension
        repeat_index = tuple([1, 1] +(scope-1)*[1] + [k])
        # Now we have a probability distribution that has the same dimension as what we have
        # at the beginning and a polished last dimension where we repeat the probability for subsequences
        # ending at the second last dimensions.
        new_prob_table = new_probability.repeat(repeat_index)
        return most_probable_word_index, new_prob_table, corresponding_prob

    def top_k_filtering(self, logits, k, prev_index_to_id_dict, prev_id_to_index_dict, blank_token_id=2):
        """
        Get the top-k most probable token and their corresponding probabilities
        logits (tensor): the logits returned by the model at the current time step
        k (int): the number of the top tokens that we desire
        blank_token_id (int): the id of the blank token

        Return:
            values (tensor of size k): the probability of the most probable tokens, with the first one set to be the blank token
            index_to_id_dict (dict of size k): a dictionary mapping from the element index of the values vector to their real id
            repeated_element_index_list (list): a list of the index (row, column) indicating the repeating index
        """

        values, ids = logits.topk(k)
        # print("pure top k values")
        # print(values)
        if blank_token_id in ids:
            # If the blank token is one of the top k token
            temp_index = ((ids == blank_token_id).nonzero(as_tuple=True)[0]).item()  # the index of the blank token
            temp_value = values[temp_index].clone()  # the prob_value of the blank token
            values[temp_index] = values[0]  # swap the index between the top token and the blank token
            ids[temp_index] = ids[0]
            values[0] = temp_value  # we place the blank token's probability as the first element of the value vector
            ids[0] = blank_token_id
            # print("blank in ids")
            # print(values)
        else:
            # If the blank token is not one of the top k tokens
            values[1:k] = values[0:k-1].clone()  # we perform a shift and drop the last top token
            ids[1:k] = ids[0:k-1].clone()
            values[0] = logits[blank_token_id]
            ids[0] = blank_token_id
            # print("blank not in ids")
            # print(values)
        index_to_id_dict = {}
        id_to_index_dict = {}
        repeated_element_index_list = []  # index of repeated elements
        non_repeated_element_non_special_index_list = [] # index of non-repeated elements

        for index in range(0, k):
            # We create a dictionary mapping from the index of each value in the values tensor to it's corresponding id
            # in the logits tensor
            current_dict_id = ids[index].item()
            index_to_id_dict[index] = current_dict_id
            id_to_index_dict[current_dict_id] = index
            if prev_index_to_id_dict is not None:
                # If we are not operating on the first dictionary
                if current_dict_id in prev_index_to_id_dict.values():
                    prev_dict_element_index = prev_id_to_index_dict[current_dict_id]
                    repeated_element_index_list.append((prev_dict_element_index, index))
        repeated_or_special_element_index_list = copy.deepcopy(repeated_element_index_list)

        all_combinations_no_blank = set(itertools.product(list(range(0, k)), list(range(1, k))))
        all_combinations_blank = set(itertools.product(list(range(0, k)), list(range(0, 1))))

        # remaining_all_combinations_no_blank = all_combinations_no_blank - set(repeated_element_index_list)
        # remaining_all_combinations_blank = all_combinations_blank - set(repeated_element_index_list)
        # non_repeated_element_non_special_index_list = list(set.union(remaining_all_combinations_no_blank, set(non_repeated_element_non_special_index_list)))
        # repeated_or_special_element_index_list = list(set.union(remaining_all_combinations_blank,set(repeated_or_special_element_index_list)))
        for i in range(0, k):
            for j in range(0,k):
                if (i, j) not in repeated_element_index_list:
                    if index_to_id_dict[j] != blank_token_id:
                        non_repeated_element_non_special_index_list.append((i, j))
                    else:
                        repeated_or_special_element_index_list.append((i, j))
        # Notice if the repeated token is not in the top-k token dictionary of the current time step, we don't include
        # it in the remove_repeated_element_mask.
        assert len(repeated_or_special_element_index_list) == len(set(repeated_or_special_element_index_list)), "there are repeated index!"
        assert len(non_repeated_element_non_special_index_list) == len(set(non_repeated_element_non_special_index_list)), "there are repeated index!"
        assert -1 not in index_to_id_dict.keys()
        assert -1 not in index_to_id_dict.values()
        repeated_or_special_element_index_list = [tuple([x[0] for x in repeated_or_special_element_index_list]),
                                                  tuple([x[1] for x in repeated_or_special_element_index_list])]
        non_repeated_element_non_special_index_list = [tuple([x[0] for x in non_repeated_element_non_special_index_list]),
                                                  tuple([x[1] for x in non_repeated_element_non_special_index_list])]
        return values, index_to_id_dict, id_to_index_dict, repeated_or_special_element_index_list, non_repeated_element_non_special_index_list


    def ctc_length_control_initialization(self, logits, desired_length, scope, k, device, path_num=-1):
        """
        Perform the initialization of the dynamic programming ctc length control algorithm
        Input:
            logits: Tensor(sequence_length*num_words) logits over different words (we force batch size to be 1 for now)
            desired_length: (int) the desired length of the summary.
            scope: (int) the length of the (ctc token) subsequence probability distribution that we are tracking
            k: (int) the number of most probable words that we keep in each position
            device: the device to store probabilities & perform calculation
        Return:
            prob_sequence: (Tensor) The probability over all possible words at each time step
            ctc_sequence_length: (int) The length of the ctc sequence
            dp_token_table: (Tensor) The determined token at each table slot
            dp_dict_table: (Tensor) The dictionary mapping from word index to word id at each table slot
            dp_prob_table: (Tensor) The table to store the probability of generating summaries of various lengths
                            given different length of the logit sequence.
            dp_marg_prob_table: (Tensor) The table to store the marginalized version of dp_prob_table
        """
        prob_sequence = torch.nn.functional.softmax(logits, dim=-1)  # Get the log probability from logits
        # Notice scope = 1 means we only consider the probability distribution over words for the current time step
        # Notice k = 10 means we only care about the top 10 words at each time step
        ctc_sequence_length = len(prob_sequence)  # The length of the ctc output sequence.
        if scope > ctc_sequence_length:
            # If the scope exceeds ctc_sequence_length
            raise ValueError("The scope to reduce cannot exceed the length of the ctc output sequence")
        elif scope < 1:
            raise ValueError("The scope must be positive integer")
        # token table dimension store the determined token at each table slot
        # dict table dimension store the dictionary mapping from index (e.g., 1~k) to token id (e.g., 1~50000)
        # prob_dimensions determines how many & how large distributions do we trace
        dp_prob_dimensions = [desired_length, ctc_sequence_length, path_num] + scope * [k]
        dp_marg_prob_dimensions = [desired_length, ctc_sequence_length, path_num] + scope * [k]
        dp_prefix_id_table = torch.zeros([desired_length, ctc_sequence_length, path_num, ctc_sequence_length],
                                         dtype=torch.long, device=device) - 1
        dp_index_to_id_dict_list = [-1]*ctc_sequence_length  # The main table to store the dictionary mapping from token index to token id
        dp_id_to_index_dict_list = [-1]*ctc_sequence_length
        dp_prob_table = torch.zeros(dp_prob_dimensions, dtype=prob_sequence.dtype,
                                    device=device)  # The main table to store DP (dynamic programming) probability result
        # We perform a minus to indicate that this table cannot be used without initialzation
        # Error would occour if the table is used without initialization.
        dp_marg_prob_table = torch.zeros(dp_marg_prob_dimensions, dtype=prob_sequence.dtype,
                                    device=device) -1  # The main table to store DP (dynamic programming) probability result
        dp_log_token_corresponding_prob_table = torch.zeros([desired_length, ctc_sequence_length, path_num], dtype=prob_sequence.dtype,
                                    device=device)
        dp_log_prefix_prob_table = torch.zeros([desired_length, ctc_sequence_length, path_num], dtype=prob_sequence.dtype,
                                    device=device)

        return prob_sequence, ctc_sequence_length, dp_prefix_id_table, dp_index_to_id_dict_list, \
               dp_id_to_index_dict_list, dp_prob_table, dp_marg_prob_table, dp_log_token_corresponding_prob_table, \
                dp_log_prefix_prob_table

    def get_special_tokens_prob(self, special_token_ids, current_index_to_id_dict, current_id_to_index_dict,
                                    current_filtered_log_prob, replacing_value=float("-inf")):
        only_special_cloned_prob = current_filtered_log_prob.clone()
        for i in current_index_to_id_dict.values():
            if i not in special_token_ids:
                token_index = current_id_to_index_dict[i]
                # add the index of the special token to the list
                only_special_cloned_prob[token_index] = replacing_value
        return only_special_cloned_prob

    def get_non_special_tokens_prob(self, special_token_ids, current_index_to_id_dict, current_id_to_index_dict,
                                current_filtered_log_prob, replacing_value=float("-inf")):
        only_non_special_cloned_prob = current_filtered_log_prob.clone()
        for i in current_index_to_id_dict.values():
            if i in special_token_ids:
                token_index = current_id_to_index_dict[i]
                # add the index of the special token to the list
                only_non_special_cloned_prob[token_index] = replacing_value
        return only_non_special_cloned_prob

    def row_inference(self, column, prev_max_row_index, reshaping_index, remaining_scope, split_point, path_num, scope, k,
                      desired_length, only_special_cloned_prob, only_non_special_cloned_prob,
                      non_special_non_repeated_transition_matrix, special_or_repeated_transition_matrix, dp_log_prob_table,
                      dp_log_prefix_prob_table, dp_prefix_id_table, prev_prob_table, chosen_token_index_list, chosen_token_id_list, test_table=None):
        """
        Perform actual table filling for each rows.
        """
        if column == 0:
            # For the first row, we initialize all non-special tokens with -inf probability
            # we only initialize the value to the first rwo to avoid repeated path
            dp_log_prob_table[0, column, 0] += only_special_cloned_prob[reshaping_index]
            dp_log_prob_table[0, column, 1:] += float("-inf")
            test_table[0,column, 0, 0] = 0
            # For the second row, we initialize it with the probability of non-special tokens
            # we only initialize the value to the first row to avoid repeated path
            dp_log_prob_table[1, column, 0] += only_non_special_cloned_prob[reshaping_index]
            dp_log_prob_table[1, column, 1:] += float("-inf")
            test_table[1, column, 0, 0] = 1
        else:
            # For other columns, we first solve the first row and the last row since they do not require case split.
            dp_log_prob_table[0, column, 0] = prev_prob_table[0, column - 1, 0] + only_special_cloned_prob[reshaping_index]
            dp_log_prob_table[0, column, 1:] = float("-inf")
            test_table[0, column, 0] = test_table[0, column-1, 0]
            test_table[0, column, 0, column] = 0
            if column + 1 < desired_length:
                # If we still have space for pure expansion
                dp_log_prob_table[column + 1, column, 0] = prev_prob_table[column, column - 1, 0] \
                                                        + non_special_non_repeated_transition_matrix[reshaping_index]
                dp_log_prob_table[column + 1, column, 1:] = float("-inf")
                test_table[column + 1, column, 0] = test_table[column, column - 1, 0]
                test_table[column + 1, column, 0, column] = column + 1
            # For other rows (i.e., middle rows)
            # We repeat the assignment for a couple times to fill up the table slot
            repeat_ratio = int(path_num / (2 * split_point))
            repeat_index = tuple([1, repeat_ratio] + scope * [1])
            if remaining_scope > 0:
                # If we still have enough position to store the paths and probabilities
                # We first determine the first half probability, which goes from diagonal-neighbouring slot.
                dp_log_prob_table[1:prev_max_row_index+1, column, 0:2*split_point] = torch.cat(
                    [prev_prob_table[:prev_max_row_index, column - 1, 0:split_point] +
                     non_special_non_repeated_transition_matrix[reshaping_index],
                     prev_prob_table[1:prev_max_row_index+1, column - 1, 0:split_point] +
                     special_or_repeated_transition_matrix[reshaping_index]],
                    dim=1)
                dp_log_prob_table[1:prev_max_row_index+1, column, 2*split_point:] = float("-inf")
                test_table[1:prev_max_row_index+1, column, 0:split_point] = test_table[:prev_max_row_index, column - 1, 0:split_point]
                test_table[1:prev_max_row_index + 1, column, split_point:2*split_point] = test_table[1:prev_max_row_index+1,
                                                                              column - 1, 0:split_point]
                test_table[1:prev_max_row_index + 1, column, 0:2*split_point, column] += torch.arange(1, prev_max_row_index + 1).unsqueeze(1) +1
            else:
                # If we are running out of space and marginalization was performed
                # We first store the best paths in previous table slots
                prev_best_path_prob, prev_best_path_index = dp_log_prefix_prob_table[:prev_max_row_index+1,
                                                            column - 1].topk(split_point, dim=1)
                # upper path probability records the probability transiting from the diagonal-neighbouring slots.
                diagonal_neighbouring_row_index = torch.arange(0, prev_max_row_index).repeat(split_point, 1).transpose(0, 1).flatten()
                row_neighbouring_row_index = torch.arange(1, prev_max_row_index+1).repeat(split_point, 1).transpose(0, 1).flatten()
                diagonal_neighbouring_prob = prev_prob_table[diagonal_neighbouring_row_index, column - 1,
                                                             prev_best_path_index[0:prev_max_row_index, :].view(-1)].view([prev_max_row_index, split_point] + scope * [k])
                # lower path probability records the probability transiting from the row-neighbouring slots.
                row_neighbouring_prob = prev_prob_table[row_neighbouring_row_index, column - 1,
                                                        prev_best_path_index[1:prev_max_row_index+1, :].view(-1)].view([prev_max_row_index, split_point] + scope * [k])
                # upper + non_special_non_reapeat
                # lower + special_or_reapeat
                dp_log_prob_table[1:prev_max_row_index+1, column] = \
                    torch.cat([diagonal_neighbouring_prob + non_special_non_repeated_transition_matrix[reshaping_index],
                               row_neighbouring_prob + special_or_repeated_transition_matrix[reshaping_index]], dim=1)

                dp_prefix_id_table[0, column] = dp_prefix_id_table[0, column - 1]
                dp_prefix_id_table[0, column, :, column] = chosen_token_id_list[0]
                if column + 1 < desired_length:
                    # If we still have pure expansion
                    dp_prefix_id_table[column + 1, column] = dp_prefix_id_table[column, column - 1]
                    dp_prefix_id_table[column + 1, column, :, column] = chosen_token_id_list[column]
                    # If it's at middle rows, it cannot perfectly inherits and therefore needs filtering

                dp_prefix_id_table[1:prev_max_row_index+1, column, 0:split_point] = \
                    dp_prefix_id_table[diagonal_neighbouring_row_index, column - 1, prev_best_path_index[0:prev_max_row_index].view(-1)].view(prev_max_row_index, split_point, -1)
                dp_prefix_id_table[1:prev_max_row_index+1, column, 0:split_point, column] = \
                    chosen_token_id_list[diagonal_neighbouring_row_index, prev_best_path_index[0:prev_max_row_index].view(-1)].view(prev_max_row_index, split_point)

                dp_prefix_id_table[1:prev_max_row_index+1, column, split_point:] = \
                    dp_prefix_id_table[row_neighbouring_row_index, column - 1, prev_best_path_index[1:prev_max_row_index+1].view(-1)].view(prev_max_row_index, split_point, -1)
                dp_prefix_id_table[1:prev_max_row_index + 1, column, split_point:, column] = \
                    chosen_token_id_list[row_neighbouring_row_index, prev_best_path_index[1:prev_max_row_index+1].view(-1)].view(prev_max_row_index, split_point)

                test_table[1:prev_max_row_index+1, column, 0:split_point] \
                    = test_table[diagonal_neighbouring_row_index, column - 1, prev_best_path_index[0:prev_max_row_index].view(-1)].view(prev_max_row_index, split_point, -1)

                test_table[1:prev_max_row_index + 1, column, split_point:2*split_point] \
                    = test_table[row_neighbouring_row_index, column - 1, prev_best_path_index[1:prev_max_row_index+1].view(-1)].view(prev_max_row_index, split_point, -1)
                test_table[1:prev_max_row_index + 1, column, 0:2*split_point, column] += torch.arange(1, prev_max_row_index + 1).unsqueeze(1) +1

                pass


    def column_inference(self, column, scope, k, current_log_prob, prev_index_to_id_dict, prev_id_to_index_dict,
                         desired_length, ctc_blank_token_id, dp_index_to_id_dict_list, dp_id_to_index_dict_list,
                         special_token_ids, replacing_value, dp_log_prob_table, marg_criteria, path_num,
                         dp_log_marg_prob_table, dp_log_prefix_prob_table, dp_prefix_id_table, test_table=None):
        """
        Perform table (prob table and prefix table) filling for a single column
        """
        remaining_scope = scope - column  # The remaining unoccupied dimension in the table slots of the previous column
        # The maximum index of the generated summary at the previous time step
        # For example, at the 4-th column, it can generate at most length-5 summary, so it's previous time step can
        # generate at most length-4 = length(column) summary, which has a corresponding index of 4
        prev_max_row_index = min(column, desired_length-1)
        if remaining_scope > 1:
            split_point = 2 ** (column - 1)
        else:
            split_point = 2 ** (scope - 2)
        # Get the filtered top-probabilities and dictionary mapping between token index and token id
        # Get the probability of the top k tokens at the current time step, also the mapping mask between the previous
        # time step and the current time step (top k tokens could be different at different time step).
        current_filtered_log_prob, current_index_to_id_dict, current_id_to_index_dict, \
        repeated_or_special_element_index_list, non_repeated_element_non_special_index_list = \
            self.top_k_filtering(current_log_prob, k, prev_index_to_id_dict, prev_id_to_index_dict,
                                 blank_token_id=ctc_blank_token_id)
        dp_index_to_id_dict_list[column] = current_index_to_id_dict  # Store the dictionary to list
        dp_id_to_index_dict_list[column] = current_id_to_index_dict
        store_token_column = column - scope  # The column to store the determined token during marginalization
        only_special_cloned_prob = self.get_special_tokens_prob(
            special_token_ids, current_index_to_id_dict, current_id_to_index_dict, current_filtered_log_prob,
            replacing_value=replacing_value)
        only_non_special_cloned_prob = self.get_non_special_tokens_prob(
            special_token_ids, current_index_to_id_dict, current_id_to_index_dict, current_filtered_log_prob,
            replacing_value=replacing_value)
        # Filter the list of repeated element indexs such that the remaining index tuples in the list doesn't not
        # have special token index as the second element
        non_special_non_repeated_transition_matrix = current_filtered_log_prob.expand(k, k).clone()
        non_special_non_repeated_transition_matrix[repeated_or_special_element_index_list] = replacing_value
        special_or_repeated_transition_matrix = current_filtered_log_prob.expand(k, k).clone()
        special_or_repeated_transition_matrix[non_repeated_element_non_special_index_list] = replacing_value
        chosen_token_index_list = torch.zeros([prev_max_row_index+1, path_num], dtype=torch.long, device=current_log_prob.device)
        chosen_token_id_list = torch.zeros([prev_max_row_index+1, path_num], dtype=torch.long, device=current_log_prob.device)
        if remaining_scope > 0:
            # If we have enough scope to store the current probability without requiring marginalization
            reshaping_index = tuple((...,) + (None,) * (remaining_scope - 1))
            prev_prob_table = dp_log_prob_table
        else:
            # If we don't have enough scope to store the current probability, we will perform marginalization on the
            # previous prob distributions such that the remaining scope becomes 1.
            chosen_token_index_list, new_prob_table, prefix_prob = \
                self.margin_over_prob_table(column, dp_log_prob_table, scope, k, criteria=marg_criteria,
                                            path_num=path_num, end_row=prev_max_row_index)
            reshaping_index = tuple((...,) + (None,) * (1 - 1))
            prev_dict = dp_index_to_id_dict_list[store_token_column]
            for i in range(0, prev_max_row_index+1):
                for j in range(0, path_num):
                    chosen_token_id_list[i, j] = (prev_dict[chosen_token_index_list[i, j].item()])
            # Add the marginalized probability to prob table
            dp_log_marg_prob_table[:prev_max_row_index+1, column - 1] = new_prob_table
            # Add the token probability to token prob table
            dp_log_prefix_prob_table[:prev_max_row_index+1, column - 1] = prefix_prob
            prev_prob_table = dp_log_marg_prob_table
        self.row_inference(column, prev_max_row_index, reshaping_index, remaining_scope, split_point, path_num, scope, k,
                      desired_length, only_special_cloned_prob, only_non_special_cloned_prob,
                      non_special_non_repeated_transition_matrix, special_or_repeated_transition_matrix, dp_log_prob_table,
                      dp_log_prefix_prob_table, dp_prefix_id_table, prev_prob_table, chosen_token_index_list, chosen_token_id_list, test_table=test_table)

    def my_remove_adjacent(self,nums):
            return [a for a, b in zip(nums, nums[1:] + [not nums[-1]]) if a != b]

    def ctc_length_control(self, logits, length_control, desired_length, scope=1, k=10, marg_criteria="mean",
                           ctc_pad_token_id=1, ctc_bos_token_id=0, ctc_eos_token_id=2, device=torch.device('cuda'),
                           ctc_unk_token_id=3, ctc_blank_token_id=4, underflow_mag_ratio=1000):
        """
        This function perform length control on the output CTC logits of the model decoder.

        Inputs:
            logits: Tensor(sequence_length*num_words) logits over different words (we force batch size to be 1 for now)
            length_control: (str) the method that we are going to adopt to perform length control.
            desired_length: (int) the desired length of the summary.
            scope: (int) the length of the (ctc token) subsequence probability distribution that we are tracking
            k: (int) the number of most probable words that we keep in each position
            marg_criteria: (string) in which criteria do we perform marginalization on old logits.
            ctc_blank_token_id: (int) the id of the ctc blank token.
            ctc_eos_token_id: (int) the id of the ctc eos token
            device: the device to store probabilities & perform calculation

        There are five different cases:
            (Notice first row represents a resulting summary of length 0, first column represents given first logits)
            1) If the current table slot is the 1-th row & 1-th column, the only possible token is [blank] token. We
                append the [blank] token to the token table to indicate that the ctc token for this table slot has been
                determined.
                Notice we set all elements in current_table_slot[0] to be the probability of this determined token and
                other elements to be 0 to indicate that the only non-zero probability happens at the [blank] token.
            2) If the current table slot is the 1-th row & i-th column (i!=1), similarly to the previous scenario,
                the only posible token is [blank] token
            3) If the current table slot is the 2-th row & 1-th column, similarly to previous scenario, we don't have
                previous prob, so we select the top k non-blank probability of from the current probability and append
                them to the corresponding elements of the current table_slot. (e.g., table_slot[1] = current_prob[1].
            4) If the current table slot is the (i+1)-th row & i-th column, we must continue from the i-th row and the
                (i-1)-th column and generate a new token which is not repeated as the token in the i-th row and the (i-1)
                -th column.
            5) If the current table slot is i-th row & j-th column, where i < j, there are two choices to pick.
                The first one is to
                continue with the subsequence ended at table slot (i-1, j-1) and generate a non-repeated token. The second
                choice is to continue with the subsequence ended at table slot (i, j-1) and generate either a blank token
                or a repeated token.
        """
        ############################################# Initialization #############################################
        desired_length += 1  # we add an extra desired length to make sure we generated summary from 0 to k.
        if length_control == "greedy_dp":
            if scope != 1:
                raise ValueError("Scope must be 1 if using greedy dynamic programming")
            raise NotImplementedError("Haven't written code for scope 1")
        path_num = 2**(scope-1)  # Number of tracked path, we make this  number deterministically related to scope
        # desired_length = round(0.34*len(logits))+1
        prob_sequence, ctc_sequence_length, dp_prefix_id_table, dp_index_to_id_dict_list, dp_id_to_index_dict_list, \
        dp_log_prob_table, dp_log_marg_prob_table, dp_log_token_corresponding_prob_table, dp_log_prefix_prob_table \
            = self.ctc_length_control_initialization(logits, desired_length, scope, k, device, path_num=path_num)
        _, naive_summary =prob_sequence.max(-1)
        naive_summary = self.my_remove_adjacent(naive_summary.tolist())
        naive_summary = list(filter((ctc_blank_token_id).__ne__, naive_summary))
        use_shorter_summary = (len(naive_summary) <= desired_length-1) and (not self.force_length)
        source_too_short = (len(prob_sequence) <= desired_length-1)  # If the source sentence/CTC output sequence does not have enough length
        if use_shorter_summary or source_too_short:
            # The untruncated summary already has a satisfied length
            return naive_summary + (desired_length-1-len(naive_summary))*[0]
        special_token_ids = [ctc_blank_token_id]  # A list of token ids that does not contribute to the real words generation
        test_table = torch.zeros([desired_length, ctc_sequence_length, path_num, ctc_sequence_length], dtype=torch.long) -1
        replacing_value = float("-inf")
        ############################################# Main Loop ##################################################
        for column in range(0, ctc_sequence_length):
            # For each column, we calculate the desired probability and prefix for each allowable table slot.
            current_log_prob = prob_sequence[column].log()
            if column == 0:
                prev_index_to_id_dict = None
                prev_id_to_index_dict = None
            else:
                prev_index_to_id_dict = dp_index_to_id_dict_list[column - 1]
                prev_id_to_index_dict = dp_id_to_index_dict_list[column - 1]
            self.column_inference(column, scope, k, current_log_prob, prev_index_to_id_dict, prev_id_to_index_dict,
                         desired_length, ctc_blank_token_id, dp_index_to_id_dict_list, dp_id_to_index_dict_list,
                         special_token_ids, replacing_value, dp_log_prob_table, marg_criteria, path_num,
                         dp_log_marg_prob_table, dp_log_prefix_prob_table, dp_prefix_id_table, test_table=test_table)


        # Now we have a partially determined ctc token, we still need to marginalize the remaining probabilities in the
        # dp probability table to fill in the whole token table.
        # The first token can be obtained from the next function so we only include the last few tokens.
        prob_table_slot = dp_log_prob_table[desired_length-1, ctc_sequence_length-1]
        maximum_index = (prob_table_slot == torch.amax(prob_table_slot)).nonzero()[0]
        postfix_sentence = maximum_index[1:]
        generated_sentence = dp_prefix_id_table[desired_length-1, ctc_sequence_length-1, maximum_index[0]].tolist()
        for i in range(0, len(postfix_sentence)):
            store_token_column = ctc_sequence_length - scope +i
            current_index = postfix_sentence[i].item()
            current_dict = dp_index_to_id_dict_list[store_token_column]
            current_id = current_dict[current_index]
            generated_sentence += [current_id]
        generated_sentence = self.my_remove_adjacent(generated_sentence)
        generated_sentence = list(filter((ctc_blank_token_id).__ne__, generated_sentence))
        generated_sentence = list(filter((-1).__ne__, generated_sentence))
        assert len(generated_sentence) == desired_length-1, "Generated summary has a wrong length"
        return generated_sentence

    def ctc_beam_search_length_control_initialization(self, logits, desired_length, device, beam_size=1):
        prob_sequence = torch.nn.functional.softmax(logits, dim=-1)  # Get the log probability from logits
        ctc_sequence_length = len(prob_sequence)  # The length of the ctc output sequence.
        prefix_index_table = torch.zeros([desired_length, ctc_sequence_length, beam_size, ctc_sequence_length],
                                         dtype=torch.long, device=device) - 1
        index_to_id_dict_list = [-1]*ctc_sequence_length  # The main table to store the dictionary mapping from token index to token id
        id_to_index_dict_list = [-1]*ctc_sequence_length
        log_prefix_prob_table = torch.zeros([desired_length, ctc_sequence_length, beam_size], dtype=prob_sequence.dtype,
                                    device=device)

        return prob_sequence, ctc_sequence_length, prefix_index_table, log_prefix_prob_table, \
               index_to_id_dict_list, id_to_index_dict_list


    def beam_search_row_inference(self, column, prev_max_row_index, beam_size, desired_length, replacing_value, k,
                                       only_special_cloned_prob, special_or_repeated_transition_matrix, non_special_non_repeated_transition_matrix,
                                       only_non_special_cloned_prob, log_prefix_prob_table, prefix_index_table):
        # prev_index is a prev_max_row*beam_size_table
        """
        Perform actual table filling for each rows.
        """
        half_beam_size = beam_size//2
        token_prob = only_non_special_cloned_prob.clone()
        token_prob[0] = only_special_cloned_prob[0]  # we build a vector that contains all top k probability
        beam_axis = torch.arange(0, beam_size, dtype=torch.long, device=log_prefix_prob_table.device)
        half_beam_axis = torch.arange(0, half_beam_size, dtype=torch.long, device=log_prefix_prob_table.device)

        # For the first row
        if column == 0:
            log_prefix_prob_table[0, column] += only_special_cloned_prob[0:beam_size]  # Given only first beam in the blank row the right prob
            log_prefix_prob_table[1, column] += only_non_special_cloned_prob[1:beam_size+1]  # This should be a list of the possible first token
            prefix_index_table[0,column,0,0] = 0
            prefix_index_table[1,column,:,0] = beam_axis + 1
            return  # we don't need any more operation for the first column
        else:
            log_prefix_prob_table[0, column] = log_prefix_prob_table[0, column - 1] + only_special_cloned_prob[0:beam_size] # Given only first beam in the blank row the right prob
            prefix_index_table[0, column, 0, 0:column] = prefix_index_table[0, column-1, 0, 0:column]
            prefix_index_table[0, column, 0, column] = 0

        # For pure expansion row
        if column + 1 < desired_length:
            # If we still have space for pure expansion
            diagonal_neighbour_prob = log_prefix_prob_table[column, column - 1]
            diagonal_neighbour_last_index = prefix_index_table[column, column - 1, :, column-1].flatten()
            new_beam_prob_diagonal = torch.zeros([beam_size, k, k], dtype=log_prefix_prob_table.dtype, device=log_prefix_prob_table.device)
            new_beam_prob_diagonal += replacing_value
            new_beam_prob_diagonal[beam_axis, diagonal_neighbour_last_index] = 0  # first initialize used rows to 0
            new_beam_prob_diagonal[beam_axis, diagonal_neighbour_last_index] += diagonal_neighbour_prob.unsqueeze(1)  # then add
            new_beam_prob_diagonal += non_special_non_repeated_transition_matrix
            top_beam_values_diagonal, top_beam_index_diagonal = new_beam_prob_diagonal.view(1, -1).topk(beam_size, dim=-1)
            # we don't run a loop since we have only one row, top index are then correct
            top_beam_index_diagonal = unravel_indices(top_beam_index_diagonal, new_beam_prob_diagonal.shape)
            log_prefix_prob_table[column+1, column] = top_beam_values_diagonal.clone()
            prefix_index_table[column+1, column, :, 0:column] = \
                 prefix_index_table[column, column-1, top_beam_index_diagonal[:,:, 0], 0:column]
            prefix_index_table[column+1, column, :, column] = top_beam_index_diagonal[:,:, 2]
            # could include more elements to avoid greedy
        if column > 0:  # If we are not at the first column, then there is middle rows.
            num_middle_rows = prev_max_row_index  # we subtract the first row
            new_beam_prob_diagonal = torch.zeros([num_middle_rows, half_beam_size, k, k], dtype=log_prefix_prob_table.dtype, device=log_prefix_prob_table.device)
            new_beam_prob_row = torch.zeros([num_middle_rows, half_beam_size, k, k], dtype=log_prefix_prob_table.dtype, device=log_prefix_prob_table.device)
            new_beam_prob_diagonal += replacing_value  # initialize all value to -inf
            new_beam_prob_row += replacing_value

            # get the prob & last index of all beams of all middle rows
            diagonal_neighbour_last_index = prefix_index_table[0:prev_max_row_index, column - 1, :, column - 1]
            row_neighbour_last_index = prefix_index_table[1:prev_max_row_index+1, column - 1, :, column - 1]
            diagonal_neighbour_prob = log_prefix_prob_table[0:prev_max_row_index, column - 1]
            row_neighbour_prob = log_prefix_prob_table[1:prev_max_row_index+1, column - 1]

            # build index so we can fill in the new_beam_prob_diagonal and new_beam_prob_row table
            diagonal_row_axis = torch.arange(0, num_middle_rows, dtype=torch.long, device=log_prefix_prob_table.device)
            neighbouring_row_axis = torch.arange(1, num_middle_rows + 1, dtype=torch.long, device=log_prefix_prob_table.device)

            # Record some index to reuse
            diagonal_prob_index = [diagonal_row_axis.view(-1,1), half_beam_axis.repeat(num_middle_rows,1), diagonal_neighbour_last_index[:,0:half_beam_size]]
            row_prob_index = [neighbouring_row_axis.view(-1,1)-1, half_beam_axis.repeat(num_middle_rows,1), row_neighbour_last_index[:,0:half_beam_size]]

            # Fill in new_beam_prob_diagonal table and new_beam_prob_row
            new_beam_prob_diagonal[diagonal_prob_index] = 0
            new_beam_prob_diagonal[diagonal_prob_index] += diagonal_neighbour_prob[:, 0:half_beam_size].unsqueeze(2)
            new_beam_prob_row[row_prob_index] = 0
            new_beam_prob_row[row_prob_index] += row_neighbour_prob[:, 0:half_beam_size].unsqueeze(2)

            new_beam_prob_diagonal[diagonal_row_axis, :] += non_special_non_repeated_transition_matrix
            new_beam_prob_row[neighbouring_row_axis-1, :] += special_or_repeated_transition_matrix

            # Select beams and process some index
            top_beam_values_diagonal, top_beam_index_diagonal = new_beam_prob_diagonal.view(num_middle_rows, -1).topk(half_beam_size, dim=-1)
            top_beam_index_diagonal = unravel_indices(top_beam_index_diagonal, new_beam_prob_diagonal[0].shape) # separately find index for each row
            # top_beam_index_diagonal = torch.stack(top_beam_index_diagonal, dim=0)  # stack result for each row together

            top_beam_values_row, top_beam_index_row = new_beam_prob_row.view(num_middle_rows, -1).topk(half_beam_size, dim=-1)
            top_beam_index_row = unravel_indices(top_beam_index_row, new_beam_prob_row[0].shape)  # separately find index for each row
            # top_beam_index_row = torch.stack(top_beam_index_row, dim=0)  # stack result for each row together

            # use row_neighbour_transitions_axis since we are indexing all middle rows in current column
            log_prefix_prob_table[neighbouring_row_axis.view(-1,1), column, half_beam_axis] = top_beam_values_diagonal

            log_prefix_prob_table[neighbouring_row_axis.view(-1,1), column, half_beam_axis + half_beam_size] = top_beam_values_row

            prefix_index_table[neighbouring_row_axis.view(-1,1), column, half_beam_axis, 0:column] = \
                prefix_index_table[diagonal_row_axis.view(-1,1), column-1, top_beam_index_diagonal[:,:,0], 0:column]
            prefix_index_table[neighbouring_row_axis.view(-1,1), column, half_beam_axis, column] = \
                top_beam_index_diagonal[:,:,2]

            prefix_index_table[neighbouring_row_axis.view(-1,1), column, half_beam_axis+half_beam_size, 0:column] = \
                prefix_index_table[neighbouring_row_axis.view(-1,1), column-1, top_beam_index_row[:,:,0], 0:column]
            prefix_index_table[neighbouring_row_axis.view(-1,1), column, half_beam_axis+half_beam_size, column]  = \
                top_beam_index_row[:,:,2]

            # resort
            sorted, indices = torch.sort(log_prefix_prob_table[1:prev_max_row_index+1,column], 1, descending=True)  # resort the so first several beam is the most probable.
            prefix_index_table[1:prev_max_row_index+1,column] = prefix_index_table[neighbouring_row_axis.view(-1,1), column, indices]
            log_prefix_prob_table[1:prev_max_row_index+1,column] = sorted


    def beam_search_column_inference(self, column, ctc_blank_token_id, replacing_value, beam_size, desired_length,
                                              current_log_prob, index_to_id_dict_list, id_to_index_dict_list,
                                              log_prefix_prob_table, prefix_index_table, k=10, test_table=None):


        """
        Perform table (prob table and prefix table) filling for a single column
        """
        if column == 0:
            prev_index_to_id_dict = None
            prev_id_to_index_dict = None
        else:
            prev_index_to_id_dict = index_to_id_dict_list[column - 1]
            prev_id_to_index_dict = id_to_index_dict_list[column - 1]

        prev_max_row_index = min(column, desired_length-1)  # Notice it's an index
        current_filtered_log_prob, current_index_to_id_dict, current_id_to_index_dict, \
        repeated_or_special_element_index_list, non_repeated_element_non_special_index_list = \
            self.top_k_filtering(current_log_prob, k, prev_index_to_id_dict, prev_id_to_index_dict,
                                 blank_token_id=ctc_blank_token_id)

        index_to_id_dict_list[column] = current_index_to_id_dict  # Store the dictionary to list
        id_to_index_dict_list[column] = current_id_to_index_dict

        only_special_cloned_prob = self.get_special_tokens_prob(
            [ctc_blank_token_id], current_index_to_id_dict, current_id_to_index_dict, current_filtered_log_prob,
            replacing_value=replacing_value)
        only_non_special_cloned_prob = self.get_non_special_tokens_prob(
            [ctc_blank_token_id], current_index_to_id_dict, current_id_to_index_dict, current_filtered_log_prob,
            replacing_value=replacing_value)

        non_special_non_repeated_transition_matrix = current_filtered_log_prob.expand(k, k).clone()
        non_special_non_repeated_transition_matrix[repeated_or_special_element_index_list] = replacing_value
        special_or_repeated_transition_matrix = current_filtered_log_prob.expand(k, k).clone()
        special_or_repeated_transition_matrix[non_repeated_element_non_special_index_list] = replacing_value

        #dp_log_prefix_prob_table[:prev_max_row_index+1, column - 1] = prefix_prob
        self.beam_search_row_inference(column, prev_max_row_index, beam_size, desired_length, replacing_value, k,
                                       only_special_cloned_prob, special_or_repeated_transition_matrix, non_special_non_repeated_transition_matrix,
                                       only_non_special_cloned_prob, log_prefix_prob_table, prefix_index_table)


    def ctc_beam_search_length_control(self, source_sentence, logits, desired_length, beam_size=3, k=10,
                           ctc_pad_token_id=1, ctc_bos_token_id=0, ctc_eos_token_id=2, device=torch.device('cuda'),
                           ctc_unk_token_id=3, ctc_blank_token_id=4, underflow_mag_ratio=1000, use_length_ratio=False):
        ############################################# Initialization #############################################
        assert beam_size>0, "Beam size are required to be positive"
        assert k > beam_size+1, "The k has to be at least one above the beam size of the algorithm"
        assert desired_length > 0, "The desired length should be greater than 0"
        assert np.mod(beam_size, 2) ==0, "The beam size must be even number"
        if use_length_ratio:
            # We set desired length based on the length of input
            self.force_length = True  # Length are forced
            desired_length = int(np.floor(0.01*desired_length*(len(logits)-3)))  # desired length is now a ratio between 0 and 1
        desired_length += 1  # we add an extra desired length to make sure we generated summary from 0 to k.

        ################################## Check whether return original summary ##################################
        _, naive_summary = logits.max(-1)
        naive_summary = self.my_remove_adjacent(naive_summary.tolist())
        naive_summary = list(filter((ctc_blank_token_id).__ne__, naive_summary))
        use_shorter_summary = (len(naive_summary) <= desired_length-1) and (not self.force_length)
        source_too_short = (len(logits) <= desired_length-1)  # If the source sentence/CTC output sequence does not have enough length
        if use_shorter_summary or source_too_short:
            # The untruncated summary already has a satisfied length
            return naive_summary + (desired_length-1-len(naive_summary))*[0]

        replacing_value = float("-inf")
        prob_sequence, ctc_sequence_length, prefix_index_table, log_prefix_prob_table, index_to_id_dict_list, \
        id_to_index_dict_list = self.ctc_beam_search_length_control_initialization(logits, desired_length, device, beam_size=beam_size)

        ############################################# Main Loop ##################################################
        for column in range(0, ctc_sequence_length):
            # For each column, we calculate the desired probability and prefix for each allowable table slot.
            current_log_prob = prob_sequence[column].log()
            self.beam_search_column_inference(column, ctc_blank_token_id, replacing_value, beam_size, desired_length,
                                              current_log_prob, index_to_id_dict_list, id_to_index_dict_list,
                                              log_prefix_prob_table, prefix_index_table, k=k)


        # Now we have a partially determined ctc token, we still need to marginalize the remaining probabilities in the
        # dp probability table to fill in the whole token table.
        # The first token can be obtained from the next function so we only include the last few tokens.

        # repeated_selection = beam_size//2
        # generated_summary = self.get_ngram_score(source_sentence, beam_size, desired_length,ctc_sequence_length,
        #                 log_prefix_prob_table, prefix_index_table, index_to_id_dict_list, ctc_blank_token_id,
        #                                          selection_scope=repeated_selection)
        max_path_index = torch.argmax(log_prefix_prob_table[desired_length-1, ctc_sequence_length-1])
        path = prefix_index_table[desired_length-1, ctc_sequence_length-1, max_path_index]
        generated_summary = []
        for i in range(0, ctc_sequence_length):
            current_dict = index_to_id_dict_list[i]
            current_token = current_dict[path[i].item()]
            generated_summary.append(current_token)

        generated_summary = self.my_remove_adjacent(generated_summary)
        generated_summary = list(filter((ctc_blank_token_id).__ne__, generated_summary))
        generated_summary = list(filter((-1).__ne__, generated_summary))
        assert len(generated_summary) == desired_length-1, "Generated summary has a wrong length"
        return generated_summary

    def get_ngram_score(self, source_sentence, beam_size, desired_length,ctc_sequence_length,
                        log_prefix_prob_table, prefix_index_table, index_to_id_dict_list, ctc_blank_token_id, selection_scope=6):
        summary_choice_list = [0]*min(beam_size,selection_scope)
        summary_choice_score_list = [0]*min(beam_size,selection_scope)
        n_list = [2,3,4]
        for i in range(0, min(beam_size,selection_scope)):
            max_path_index = torch.argmax(log_prefix_prob_table[desired_length-1, ctc_sequence_length-1])
            path = prefix_index_table[desired_length-1, ctc_sequence_length-1, max_path_index]
            generated_summary = []
            for j in range(0, ctc_sequence_length):
                current_dict = index_to_id_dict_list[j]
                current_token = current_dict[path[j].item()]
                generated_summary.append(current_token)
            generated_summary = self.my_remove_adjacent(generated_summary)
            generated_summary = list(filter((ctc_blank_token_id).__ne__, generated_summary))
            generated_summary = list(filter((-1).__ne__, generated_summary))
            summary_choice_list[i] = generated_summary
            log_prefix_prob_table[desired_length - 1, ctc_sequence_length - 1, max_path_index] = float("-inf")  # remove this choice
            for n in n_list:
                summary_choice_score_list[i] += get_n_gram_counts(source_sentence[0, 1:-1].tolist(), generated_summary, n)
        highest_score_index = summary_choice_score_list.index(max(summary_choice_score_list))
        generated_summary = summary_choice_list[highest_score_index]
        return generated_summary


    def ctc_forward_backward_length_control_initialization(self, logits, desired_length, device):
        prob_sequence = torch.nn.functional.softmax(logits, dim=-1)  # Get the log probability from logits
        ctc_sequence_length = len(prob_sequence)  # The length of the ctc output sequence.
        dp_log_prob_table = torch.zeros([desired_length, ctc_sequence_length], dtype=prob_sequence.dtype,
                                    device=device)  # The main table to store DP (dynamic programming) probability result
        dp_sequence_table = torch.zeros([desired_length, ctc_sequence_length, ctc_sequence_length], dtype=torch.long,
                                    device=device) -1
        return prob_sequence, ctc_sequence_length, dp_log_prob_table, dp_sequence_table


    def get_non_br_and_br_token_and_prob(self, column, current_log_prob, ctc_blank_token_id, dp_sequence_table, desired_length):

        maximum_row = min(desired_length, column+1)
        if column == 0:
            last_word_sequence = dp_sequence_table[:maximum_row, 0, 0]  # If this is the first column, any word can be regarded as new word
        else:
            last_word_sequence = dp_sequence_table[:maximum_row, column - 1, column - 1]

        most_probable_values, most_probable_indices = current_log_prob.topk(3)
        if most_probable_indices[0] == ctc_blank_token_id:
            # If the probability of the blank token is the highest, we can use 1 as first choice and 2 as alternative
            # First find the index of prev tokens that are the same as the most probable non-blank id
            prev_equal_1_index = (last_word_sequence == most_probable_indices[1]).nonzero(as_tuple=True)[0]
            most_probable_non_repeat_no_blank_tokens = torch.zeros([maximum_row], dtype=torch.long, device=current_log_prob.device) -1
            most_probable_non_repeat_no_blank_prob = torch.zeros([maximum_row], dtype=current_log_prob.dtype, device=current_log_prob.device) -1
            most_probable_non_repeat_no_blank_tokens[:] = most_probable_indices[1]
            most_probable_non_repeat_no_blank_prob[:] = most_probable_values[1]
            # Then replace the most_probable_non_blank_non_repeat_token and the corresponding prob in these positions by
            # our alternative.
            most_probable_non_repeat_no_blank_tokens[prev_equal_1_index] = most_probable_indices[2]
            most_probable_non_repeat_no_blank_prob[prev_equal_1_index] = most_probable_values[2]
        elif most_probable_indices[1] == ctc_blank_token_id:
            # the probability of the blank token is the second highest, we can use 0 as first choice and 2 as alternative
            prev_equal_1_index = (last_word_sequence == most_probable_indices[0]).nonzero(as_tuple=True)[0]
            most_probable_non_repeat_no_blank_tokens = torch.zeros([maximum_row], dtype=torch.long, device=current_log_prob.device) -1
            most_probable_non_repeat_no_blank_prob = torch.zeros([maximum_row], dtype=current_log_prob.dtype, device=current_log_prob.device) -1

            most_probable_non_repeat_no_blank_tokens[:] = most_probable_indices[0]
            most_probable_non_repeat_no_blank_prob[:] = most_probable_values[0]
            # Then replace the most_probable_non_blank_non_repeat_token and the corresponding prob in these positions by
            # our alternative.
            most_probable_non_repeat_no_blank_tokens[prev_equal_1_index] = most_probable_indices[2]
            most_probable_non_repeat_no_blank_prob[prev_equal_1_index] = most_probable_values[2]
        else:
            # If the top 2 token does not contain blank, we can use 0 as first choice and 1 as alternative
            prev_equal_1_index = (last_word_sequence == most_probable_indices[0]).nonzero(as_tuple=True)[0]
            most_probable_non_repeat_no_blank_tokens = torch.zeros([maximum_row], dtype=torch.long, device=current_log_prob.device) -1
            most_probable_non_repeat_no_blank_prob = torch.zeros([maximum_row], dtype=current_log_prob.dtype, device=current_log_prob.device) -1

            most_probable_non_repeat_no_blank_tokens[:] = most_probable_indices[0]
            most_probable_non_repeat_no_blank_prob[:] = most_probable_values[0]
            # Then replace the most_probable_non_blank_non_repeat_token and the corresponding prob in these positions by
            # our alternative.
            most_probable_non_repeat_no_blank_tokens[prev_equal_1_index] = most_probable_indices[1]
            most_probable_non_repeat_no_blank_prob[prev_equal_1_index] = most_probable_values[1]

        if column == 0:
            # Then is only repeat prob
            repeat_no_blank_tokens = torch.zeros([maximum_row], dtype=torch.long, device=current_log_prob.device) + ctc_blank_token_id
            repeat_no_blank_probs = torch.zeros([maximum_row], dtype=current_log_prob.dtype,
                                                device=current_log_prob.device) + current_log_prob[ctc_blank_token_id]
        else:
            # First column stands for a choice of blank, second column stands for the choice of repeat token
            repeat_no_blank_tokens = torch.zeros([maximum_row, 2], dtype=torch.long, device=current_log_prob.device) - 1
            repeat_no_blank_probs = torch.zeros([maximum_row, 2], dtype=current_log_prob.dtype, device=current_log_prob.device) - 1
            # try:
            repeat_no_blank_tokens[:, 0] = ctc_blank_token_id
            repeat_no_blank_tokens[:, 1] = last_word_sequence
            # except:
            #     pass
            repeat_no_blank_probs[:, 0] = current_log_prob[ctc_blank_token_id]
            repeat_no_blank_probs[:, 1] = current_log_prob[last_word_sequence]
            # argmax over the second dimension since we store blank in first column and repeat on the second column
            max_prob_index = repeat_no_blank_probs.argmax(dim=1)
            # Save the most probable token ahd their corresponding probs
            repeat_no_blank_tokens = repeat_no_blank_tokens[torch.arange(0, maximum_row), max_prob_index]
            repeat_no_blank_probs = repeat_no_blank_probs[torch.arange(0, maximum_row), max_prob_index]

        return most_probable_non_repeat_no_blank_tokens, most_probable_non_repeat_no_blank_prob, repeat_no_blank_tokens, repeat_no_blank_probs

    def ctc_forward_backward_column_inference(self, column, current_log_prob, desired_length, ctc_blank_token_id, replacing_value,
                         dp_log_prob_table, dp_sequence_table):
        """
        Perform table (prob table and prefix table) filling for a single column
        """
        most_probable_non_repeat_no_blank_tokens, most_probable_non_repeat_no_blank_prob, repeat_no_blank_tokens, \
        repeat_no_blank_probs = self.get_non_br_and_br_token_and_prob(column, current_log_prob, ctc_blank_token_id, dp_sequence_table, desired_length)

        # For first row, we can only assign blank token
        dp_log_prob_table[0, column] = current_log_prob[ctc_blank_token_id]
        if column != 0:
            # If it's not the first column, we inherits the probability and prev sequence of the prev-neighbouring slot
            dp_log_prob_table[0, column] += dp_log_prob_table[0, column - 1]
            dp_sequence_table[0, column, 0:column] = dp_sequence_table[0, column-1, 0:column].clone()
        dp_sequence_table[0, column, column] = ctc_blank_token_id
        # For pure expansion row
        if column + 1 < desired_length:
            dp_log_prob_table[column + 1, column] = most_probable_non_repeat_no_blank_prob[-1]  # Assign the probability of the current word to the prob matrix
            if column != 0:
                # Add the previously accumulated probability to the current table slot
                dp_log_prob_table[column + 1, column] += dp_log_prob_table[column, column - 1]
                dp_sequence_table[column + 1, column, :column] = dp_sequence_table[column, column - 1, :column].clone()
            dp_sequence_table[column + 1, column, column] = most_probable_non_repeat_no_blank_tokens[-1]
        middle_rows_num = min(column, desired_length-1)
        end_row = middle_rows_num+1
        if middle_rows_num > 0:
            # If we have some middle rows where each row has a choice to inherit one of the two previous rows.
            middle_rows_prob_matrix = torch.zeros([middle_rows_num, 2], dtype=current_log_prob.dtype, device=current_log_prob.device)
            middle_rows_prob_matrix[:, 0] = dp_log_prob_table[0:end_row-1, column-1].clone()  # diagonal-neighbouring
            middle_rows_prob_matrix[:, 1] = dp_log_prob_table[1:end_row, column-1].clone()  # row-neighbouring
            middle_rows_prob_matrix[:, 0] += most_probable_non_repeat_no_blank_prob[1:]  # skip the first row since it has been updated.
            middle_rows_prob_matrix[:, 1] += repeat_no_blank_probs[1:]
            # Now we find the most probable previous state assignment
            selected_prob_prev_rows,  most_probable_prev_rows = middle_rows_prob_matrix.max(1)  # Find the prob of the most probable choice for each row
            non_blank_no_repeat_index = (most_probable_prev_rows == 0).nonzero(as_tuple=True)[0]+1  # Add one since we skipped the first row
            blank_or_repeat_index = (most_probable_prev_rows == 1).nonzero(as_tuple=True)[0]+1
            dp_log_prob_table[1:end_row, column] = selected_prob_prev_rows
            # update the prob table for the row that chooses diagonal-neighbouring rows
            # dp_log_prob_table[non_blank_no_repeat_index, column] += dp_log_prob_table[non_blank_no_repeat_index-1, column]
            dp_sequence_table[non_blank_no_repeat_index, column, 0:column] = dp_sequence_table[non_blank_no_repeat_index-1, column-1, 0:column]
            # update the prob table for the row that chooses row-neighbouring rows
            # dp_log_prob_table[blank_or_repeat_index, column] += dp_log_prob_table[blank_or_repeat_index, column]
            dp_sequence_table[blank_or_repeat_index, column, 0:column] = dp_sequence_table[blank_or_repeat_index, column-1, 0:column]

            #assert most_probable_non_blank_token.item() >=0
            if len(non_blank_no_repeat_index) > 0:
                dp_sequence_table[non_blank_no_repeat_index, column, column] = most_probable_non_repeat_no_blank_tokens[non_blank_no_repeat_index]
            if len(blank_or_repeat_index) > 0:
                dp_sequence_table[blank_or_repeat_index, column, column] = repeat_no_blank_tokens[blank_or_repeat_index]
            pass

    def ctc_forward_backward_length_control(self, logits, length_control, desired_length, ctc_pad_token_id=1, ctc_bos_token_id=0,
                           ctc_eos_token_id=2, device=torch.device('cuda'), ctc_unk_token_id=3, ctc_blank_token_id=4,
                           underflow_mag_ratio=1000):
        desired_length += 1  # we add an extra desired length to make sure we generated summary from 0 to k.
        logits = logits.float()
        prob_sequence, ctc_sequence_length, dp_log_prob_table, dp_sequence_table \
            = self.ctc_forward_backward_length_control_initialization(logits, desired_length, device)
        inv_idx = torch.arange(logits.size(0) - 1, -1, -1, device=device).long()
        inv_logits = logits.index_select(0, inv_idx).clone()
        inv_prob_sequence, inv_ctc_sequence_length, inv_dp_log_prob_table, inv_dp_sequence_table \
            = self.ctc_forward_backward_length_control_initialization(inv_logits, desired_length, device)
        _, naive_summary =prob_sequence.max(-1)
        naive_summary_prob = prob_sequence[torch.arange(0, ctc_sequence_length), naive_summary].log().sum()
        naive_summary = naive_summary.tolist()
        naive_summary = self.my_remove_adjacent(naive_summary)
        naive_summary = list(filter((ctc_blank_token_id).__ne__, naive_summary))

        use_shorter_summary = (len(naive_summary) <= desired_length-1) and (not self.force_length)
        source_too_short = (len(prob_sequence) <= desired_length-1)  # If the source sentence/CTC output sequence does not have enough length
        if use_shorter_summary or source_too_short:
            # The untruncated summary already has a satisfied length
            return naive_summary + (desired_length-1-len(naive_summary))*[0]

        replacing_value = float("-inf")
        ############################################# Main Loop ##################################################
        for column in range(0, ctc_sequence_length):
            # For each column, we calculate the desired probability and prefix for each allowable table slot.
            current_log_prob = prob_sequence[column].log()
            self.ctc_forward_backward_column_inference(column, current_log_prob, desired_length, ctc_blank_token_id,
                        replacing_value, dp_log_prob_table, dp_sequence_table)
            current_inv_log_prob = inv_prob_sequence[column].log()
            self.ctc_forward_backward_column_inference(column, current_inv_log_prob, desired_length, ctc_blank_token_id,
                        replacing_value, inv_dp_log_prob_table, inv_dp_sequence_table)

        inv_idx_0 = torch.arange(desired_length - 1, -1, -1, device=device).long()
        inv_idx_1 = torch.arange(ctc_sequence_length - 1, -1, -1, device=device).long()
        inv_dp_log_prob_table_temp = inv_dp_log_prob_table.index_select(0, inv_idx_0).clone()
        inv_dp_log_prob_table_temp = inv_dp_log_prob_table_temp.index_select(1, inv_idx_1).clone()

        # We create a array to store the probability of all possible combinations,
        # We have ctc_sequence_length - 1 possible middle combinations (e.g., combine t_1:1 and t_2:6 if ctc_sequence_length = 6).
        # The first and last position in middle_prob_sequence goes to pure forward (i.e., t_1:T) and pure backward (i.e., t_T:1) respectively.
        middle_prob_sequence = torch.zeros([desired_length, ctc_sequence_length+1], dtype=prob_sequence.dtype, device=device)
        dp_log_prob_table[dp_log_prob_table == 0] = replacing_value
        inv_dp_log_prob_table_temp[inv_dp_log_prob_table_temp == 0] = replacing_value  # unafford positions are replaced by -inf to avoid mis count

        middle_prob_sequence[:, 1:-1] = dp_log_prob_table[:, :-1] + inv_dp_log_prob_table_temp[:, 1:]
        middle_prob_sequence[:, 0] = dp_log_prob_table[-1, -1]
        middle_prob_sequence[:, -1] = inv_dp_log_prob_table_temp[0, 0]

        max_index = (middle_prob_sequence==torch.max(middle_prob_sequence)).nonzero()[0]
        good_to_go = False
        while not good_to_go:
            if max_index[1] == 0:
                # The result of the forward-pass gives the optimum result
                generated_sentence = dp_sequence_table[desired_length - 1, ctc_sequence_length - 1].tolist()
            elif max_index[1] == ctc_sequence_length:
                generated_sentence = reversed(inv_dp_sequence_table[desired_length - 1, ctc_sequence_length - 1]).tolist()
            else:
                forward_len = max_index[0]
                backward_len = desired_length - forward_len -1
                forward_logit_len = max_index[1]
                backward_logit_len = ctc_sequence_length - forward_logit_len -1
                generated_sentence = dp_sequence_table[tuple(max_index.tolist())][:forward_logit_len].tolist() \
                                         + reversed(inv_dp_sequence_table[backward_len, backward_logit_len][:backward_logit_len+1]).tolist()
            test_sentence = self.my_remove_adjacent(generated_sentence)
            test_sentence = list(filter((ctc_blank_token_id).__ne__, test_sentence))
            test_sentence = list(filter((-1).__ne__, test_sentence))
            if len(test_sentence) == desired_length-1:
                good_to_go = True
                    # print("Good to go")
            else:
                middle_prob_sequence[tuple(max_index)] = replacing_value  # remove this choice
                max_index = (middle_prob_sequence == torch.max(middle_prob_sequence)).nonzero()[0]
                # print("retry")
        generated_sentence_prob = prob_sequence[torch.arange(0, ctc_sequence_length), generated_sentence].log().sum()
        generated_sentence = self.my_remove_adjacent(generated_sentence)
        generated_sentence = list(filter((ctc_blank_token_id).__ne__, generated_sentence))
        generated_sentence = list(filter((-1).__ne__, generated_sentence))

        assert len(generated_sentence) == desired_length-1, "Generated summary has a wrong length"
        return generated_sentence

    def forward_decoder(self, source_sentence, decoder_out, encoder_out, decoding_format=None, **kwargs):
        # set CTC decoder beam size
        # try:
        #     self.ctc_decoder._beam_width = self.args.ctc_beam_size
        # except:
        #     pass

        step = decoder_out.step
        output_tokens = decoder_out.output_tokens

        history = decoder_out.history

        # execute the decoder
            # If we are not using bert as the encoder, we decode as usual
        output_logits = self.decoder(
                normalize=False,
                prev_output_tokens=output_tokens,
                encoder_out=encoder_out,
                step=step,
        )
        if self.length_control == "greedy_dp" or self.length_control == "greedy_dp_scope_search":
            output_tokens = []
            for i in range(0, len(output_logits)):
                current_sentence = self.ctc_length_control(output_logits[i], self.length_control, self.desired_length, scope=self.scope,
                                                 k=self.k, marg_criteria=self.marg_criteria, ctc_pad_token_id=self.pad, ctc_bos_token_id=self.bos,
                                                 ctc_eos_token_id=self.eos, ctc_unk_token_id=self.unk)

                current_sentence_tensor = torch.tensor(current_sentence, dtype=torch.long, device=output_logits.device)
                output_tokens.append(current_sentence_tensor)
            output_tokens = torch.stack(output_tokens, dim=0)
            if history is not None:
                history.append(output_tokens.clone())
            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=torch.full(output_tokens.size(), 1.0),
                attn=None,
                history=history,
            )
        elif self.length_control == "forward_backward":
            output_tokens = []
            for i in range(0, len(output_logits)):
                current_sentence = self.ctc_forward_backward_length_control(output_logits[i], self.length_control, self.desired_length,
                                                                            ctc_pad_token_id=self.pad, ctc_bos_token_id=self.bos,
                                                 ctc_eos_token_id=self.eos, ctc_unk_token_id=self.unk)

                current_sentence_tensor = torch.tensor(current_sentence, dtype=torch.long, device=output_logits.device)
                output_tokens.append(current_sentence_tensor)
            output_tokens = torch.stack(output_tokens, dim=0)
            if history is not None:
                history.append(output_tokens.clone())
            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=torch.full(output_tokens.size(), 1.0),
                attn=None,
                history=history,
            )
        elif self.length_control == "beam_search_length_control":
            output_tokens = []
            for i in range(0, len(output_logits)):
                current_sentence = self.ctc_beam_search_length_control(source_sentence, output_logits[i], self.desired_length, beam_size=self.beam_size, k=self.k,
                                                                            ctc_pad_token_id=self.pad, ctc_bos_token_id=self.bos,
                                                 ctc_eos_token_id=self.eos, ctc_unk_token_id=self.unk, use_length_ratio=self.use_length_ratio)

                current_sentence_tensor = torch.tensor(current_sentence, dtype=torch.long, device=output_logits.device)
                output_tokens.append(current_sentence_tensor)
            output_tokens = torch.stack(output_tokens, dim=0)
            if history is not None:
                history.append(output_tokens.clone())
            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=torch.full(output_tokens.size(), 1.0),
                attn=None,
                history=history,
            )
        elif self.custom_ctc:
            output_scores = decoder_out.output_scores
            _scores, _tokens = output_logits.max(-1)
            output_masks = output_tokens.ne(self.pad)
            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])
            if history is not None:
                history.append(output_tokens.clone())

            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=output_scores,
                attn=None,
                history=history,
            )

        elif self.plain_ctc:
            output_scores = decoder_out.output_scores
            _scores, _tokens = output_logits.max(-1)
            output_masks = output_tokens.ne(self.pad)
            try:
                output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
                output_scores.masked_scatter_(output_masks, _scores[output_masks])
            except:
                output_tokens = _tokens
                output_scores = _scores
            if history is not None:
                history.append(output_tokens.clone())

            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=output_scores,
                attn=None,
                history=history,
            )
        else:
            # _scores, _tokens = F.log_softmax(output_logits, -1).max(-1)
            # _scores == beam_results[:,0,:]
            output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)
            beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(F.softmax(output_logits, -1),
                                                                                     output_length)
            top_beam_tokens = beam_results[:, 0, :]
            top_beam_len = out_lens[:, 0]
            mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len). \
                repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
            top_beam_tokens[~mask] = self.decoder.dictionary.pad()
            # output_scores.masked_scatter_(output_masks, _scores[output_masks])
            if history is not None:
                history.append(output_tokens.clone())

            return decoder_out._replace(
                output_tokens=top_beam_tokens.to(output_logits.device),
                output_scores=torch.full(top_beam_tokens.size(), 1.0),
                attn=None,
                history=history,
            )

    def get_search_results(self, decoder_out, encoder_out, beam_size=None, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        history = decoder_out.history
        # Set ctc beam size
        if beam_size is not None:
            self.ctc_decoder._beam_width = beam_size
        else:
            beam_size = self.ctc_decoder._beam_width

        # execute the decoder
        output_logits = self.decoder(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )
        # _scores, _tokens = F.log_softmax(output_logits, -1).max(-1)
        # _scores == beam_results[:,0,:]
        output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)
        beam_results, beam_scores, timesteps, out_lens = self.ctc_decoder.decode(F.softmax(output_logits, -1),
                                                                                 output_length)

        beam_results = beam_results[:, :, :out_lens.max()]
        beam_size = beam_scores.size(1)
        for beam_idx in range(beam_size):
            top_beam_tokens = beam_results[:, beam_idx, :]
            top_beam_len = out_lens[:, beam_idx]
            mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len). \
                repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
            top_beam_tokens[~mask] = self.decoder.dictionary.pad()
        return beam_results, beam_scores

    def initialize_output_tokens_by_src_tokens(self, src_tokens):
        if not self.copy_src_token:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            if self.args.src_upsample_scale > 2:
                length_tgt = length_tgt * self.args.src_upsample_scale
            else:
                length_tgt = length_tgt * self.args.src_upsample_scale  # + 10
            max_length = length_tgt.clamp_(min=2).max()
            idx_length = utils.new_arange(src_tokens, max_length)

            initial_output_tokens = src_tokens.new_zeros(
                src_tokens.size(0), max_length
            ).fill_(self.pad)
            initial_output_tokens.masked_fill_(
                idx_length[None, :] < length_tgt[:, None], self.unk
            )
            initial_output_tokens[:, 0] = self.bos
            initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
            return initial_output_tokens
        else:
            if self.args.src_upsample_scale <= 1:
                return src_tokens

            def _us(x, s):
                B = x.size(0)
                _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
                return _x

            return _us(src_tokens, self.args.src_upsample_scale)

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        initial_output_tokens = self.initialize_output_tokens_by_src_tokens(src_tokens)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
                length_tgt[:, None]
                + utils.new_arange(length_tgt, 1, beam_size)
                - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )


class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        def weights_init(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(
                    m.weight, mean=0, std=self.output_embed_dim ** -0.5
                )
        # torch.nn.init.normal_(
        #         self.encoder_output_layer.weight, mean=0, std=self.output_embed_dim ** -0.5
        #     )
        self.encoder_embed_dim = args.encoder_embed_dim
        self.sg_length_pred = getattr(args, "sg_length_pred", False)
        self.pred_length_offset = getattr(args, "pred_length_offset", False)
        self.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)

        self.embed_length = Embedding(256, self.encoder_embed_dim, None)
        self.softcopy = getattr(args, "softcopy", False)
        if self.softcopy:
            self.softcopy_learnable = self.args.softcopy_temp == 0
            if self.softcopy_learnable:
                self.para_softcopy_temp = torch.nn.Parameter(torch.tensor(1.0))
        else:
            self.softcopy_learnable = False

    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        decoder_out = self.output_projection(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out

    @ensemble_decoder
    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        if self.sg_length_pred:
            enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            early_exit=None,
            embedding_copy=False,
            **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if embedding_copy or self.softcopy:
            src_embd = encoder_out["encoder_embedding"][0]
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None
            src_mask = (
                ~src_mask
                if src_mask is not None
                else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
            )

            if not self.softcopy:
                x, decoder_padding_mask = self.forward_embedding(
                    prev_output_tokens,
                    self.forward_copying_source(
                        src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
                    ),
                )
            else:
                x = self.forward_softcopying_source(src_embd, src_mask, prev_output_tokens.ne(self.padding_idx))
                decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)

        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        # for i, layer in enumerate(self.layers):
        #
        #     # early exit from the decoder.
        #     if (early_exit is not None) and (i >= early_exit):
        #         break
        #
        #     x, attn, _ = layer(
        #         x,
        #         encoder_out["encoder_out"][0]
        #         if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
        #         else None,
        #         encoder_out["encoder_padding_mask"][0]
        #         if (
        #                 encoder_out is not None
        #                 and len(encoder_out["encoder_padding_mask"]) > 0
        #         )
        #         else None,
        #         self_attn_mask=None,
        #         self_attn_padding_mask=decoder_padding_mask,
        #     )
        #     inner_states.append(x)

        # if self.layer_norm:
        #     x = self.layer_norm(x)
        #
        # # T x B x C -> B x T x C
        x = encoder_out["encoder_out"][0]
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding

    def forward_softcopying_source(self, src_embeds, src_masks, tgt_masks):
        # length_sources = torch.randint(1, 26, (src_embeds.size(0), )).to(src_embeds) # src_masks.sum(1)
        # length_targets = torch.randint(1, 52, (src_embeds.size(0), )).to(src_embeds) # tgt_masks.sum(1)
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        src_len_mat = torch.div(
            (torch.arange(src_embeds.size(1), device=src_embeds.device, dtype=src_embeds.dtype)).unsqueeze(
                0).repeat(src_embeds.size(0), 1), length_sources.unsqueeze(1))
        tgt_len_mat = torch.div(
            (torch.arange(tgt_masks.size(1), device=src_embeds.device, dtype=src_embeds.dtype)).unsqueeze(
                0).repeat(src_embeds.size(0), 1), length_targets.unsqueeze(1))
        # test_sum = torch.relu(torch.einsum('km,kn->kmn', tgt_len_mat, -src_len_mat))
        # k = src_len_mat.size(0)
        m = src_len_mat.size(1)
        n = tgt_len_mat.size(1)
        # test_sum2 = torch.zeros(k, n, m)
        # for _k in range(k):
        #     for _n in range(n):
        #         for _m in range(m):
        #             test_sum2[_k, _n, _m] = torch.abs(tgt_len_mat[_k, _n] - src_len_mat[_k, _m])
        test_sum3 = - torch.abs(tgt_len_mat.unsqueeze(2).repeat(1, 1, m) - src_len_mat.unsqueeze(1).repeat(1, n, 1))
        # src_mask_2 = torch.arange(src_embeds.size(1)).expand(src_embeds.size(0), src_embeds.size(1)).to(length_sources) < length_sources.unsqueeze(1)
        test_sum3_2 = test_sum3.masked_fill(~src_masks.unsqueeze(1), -float("Inf"))
        if not self.softcopy_learnable:
            src_weight = torch.softmax(test_sum3_2 * self.args.softcopy_temp, dim=2)
        else:
            src_weight = torch.softmax(test_sum3_2 * self.para_softcopy_temp, dim=2)
        copied_embedding = torch.bmm(src_weight, src_embeds)

        return copied_embedding

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        if self.pred_length_offset:
            if src_masks is None:
                src_lengs = enc_feats.new_ones(enc_feats.size(1)).fill_(
                    enc_feats.size(0)
                )
            else:
                src_lengs = (~src_masks).transpose(0, 1).type_as(enc_feats).sum(0)
            src_lengs = src_lengs.long()

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            if self.pred_length_offset:
                length_tgt = tgt_lengs - src_lengs + 128
            else:
                length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            if self.pred_length_offset:
                length_tgt = pred_lengs - 128 + src_lengs
            else:
                length_tgt = pred_lengs

        return length_tgt


@register_model_architecture(
    "nat_encoder_only_customized_ctc", "nat_encoder_only_customized_ctc"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)  # NOTE
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)  # NOTE
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "nat_encoder_only_customized_ctc", "nat_encoder_only_customized_ctc_fixlen"
)
def base_architecture1(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)  # NOTE
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)  # NOTE
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "nat_encoder_only_customized_ctc", "nat_encoder_only_customized_ctc_refine"
)
def base_architecture2(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)  # NOTE
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)  # NOTE
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
