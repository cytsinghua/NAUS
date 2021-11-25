#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable
from pythonrouge.pythonrouge import Pythonrouge
import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm
import random
import time
from matplotlib import pyplot as plt
import re
import json
import wandb

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(cfg: DictConfig) -> None:
    # wandb.init(project="nas", settings=wandb.Settings(start_method="fork"))  # use on CC
    wandb.init(mode="disabled")
    wandb.run.name = cfg.task.save_dir
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    if "bert" in cfg.task.arch:
        # If we are replacing the encoder by bert model, we have to change eos and bos
        task.src_dict.bos_index = 101
        task.tgt_dict.bos_index = 101
        task.src_dict.eos_index = 102
        task.tgt_dict.eos_index = 102
        task.src_dict.use_bert_weight = True  # so we can control eos and bos at dataset functions
    else:
        task.src_dict.use_bert_weight = False
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per GPU = {} and batch size per GPU = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()

    def decode(toks, escape_unk=False):
        s = trainer.task.tgt_dict.string(
            toks.int().cpu(),
            trainer.task.args.eval_bleu_remove_bpe,
            unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
        )
        if trainer.task.tokenizer:
            s = trainer.task.tokenizer.decode(s)
        return s
    trainer.model.set_decode_function(decode)

    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))
    # if cfg.task.use_bert_tokens:
    #     # If we are using bert as the encoder for the training, we have to map the fairseq tokenized sequence to the bert
    #     # tokenized sequence
    #     print("converting samples...")
    #     if cfg.task.data == 'data-bin/gigaword_8_ref_index/':
    #         source_data_dir = "gigaword_8_index"
    #     elif cfg.task.data == 'data-bin/gigaword_10_ref_index/':
    #         source_data_dir = "gigaword_10_index"
    #     else:
    #         raise NotImplementedError
        # for dataset_name in trainer.task.datasets:
        #     # We perform the update for each dataset (e.g., train, valid, test)
        #     current_data_set_path = os.path.join(source_data_dir, dataset_name)
        #     current_modified_source_dataset = []
        #     current_modified_target_dataset = []
        #     sizes = []  # We also need to update the stored sizes in the dataset
        #     src_sizes = []
        #     tgt_sizes = []
        #     with open(current_data_set_path+".article") as f:
        #         current_article = f.readlines()
        #     with open(current_data_set_path+".summary") as f:
        #         current_summaries = f.readlines()
        #     num_samples = len(current_article)
        #     for i in tqdm(range(0, num_samples)):
        #         current_modified_source_sentence = current_article[i].rstrip()
        #         current_modified_source_sentence = current_modified_source_sentence.split()
        #         decoded_source = [int(x) for x in current_modified_source_sentence]
        #         decoded_source = torch.tensor(decoded_source)
        #
        #         current_modified_target_sentence = current_summaries[i].rstrip()
        #         current_modified_target_sentence = current_modified_target_sentence.split()
        #         decoded_target = [int(x) for x in current_modified_target_sentence]
        #         decoded_target = torch.tensor(decoded_target)
        #
        #         current_src_size = len(decoded_source) +1
        #         current_tgt_size = len(decoded_target)
        #         sizes.append([current_src_size,current_tgt_size])
        #         src_sizes.append(current_src_size)
        #         tgt_sizes.append(current_tgt_size)
        #         current_modified_source_dataset.append(decoded_source)
        #         current_modified_target_dataset.append(decoded_target)
        #
        #     trainer.task.datasets[dataset_name].src.dataset = current_modified_source_dataset
        #     trainer.task.datasets[dataset_name].tgt.dataset = current_modified_target_dataset
        #     trainer.task.datasets[dataset_name].sizes = np.array(sizes)
        #     trainer.task.datasets[dataset_name].src_sizes = np.array(src_sizes)
        #     trainer.task.datasets[dataset_name].tgt_sizes = np.array(tgt_sizes)
        #     # change it to -1 to check whether it will be used by fairseq
        #     trainer.task.datasets[dataset_name].src.sizes[:] = -1
        #     trainer.task.datasets[dataset_name].tgt.sizes[:] = -1
        #     pass
    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )
        with metrics.aggregate(new_root=True) as agg:
            if valid_losses[0] is None:
                pass
                #agg["rouge"] = 0
            else:
                pass
                #agg["rouge"] = valid_losses[0]

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats

def get_duc2004_ref_summary():
    # Build duc 2004 reference summary dataset from disk
    with open("DUC2004/task1_ref0.txt", 'r') as f:
        ref_1 = f.readlines()
    with open("DUC2004/task1_ref1.txt", 'r') as f:
        ref_2 = f.readlines()
    with open("DUC2004/task1_ref2.txt", 'r') as f:
        ref_3 = f.readlines()
    with open("DUC2004/task1_ref3.txt", 'r') as f:
        ref_4 = f.readlines()

    ref_1 = [sentence.rstrip() for sentence in ref_1]
    ref_2 = [sentence.rstrip() for sentence in ref_2]
    ref_3 = [sentence.rstrip() for sentence in ref_3]
    ref_4 = [sentence.rstrip() for sentence in ref_4]

    all_references = []
    for i in range(0, len(ref_1)):
        current_ref = [[ref_1[i]], [ref_2[i]], [ref_3[i]], [ref_4[i]]]
        all_references.append(current_ref)

    return all_references


def replace_words(s, words):
    """
    Post-processing of bert outputs
    """
    for k, v in words.items():
        s = s.replace(k.lower(), v)
    s = s.replace("'s", " 's")
    s = s.replace("[CLS]", "")
    s = s.replace("[PAD]", "")
    s = s.replace("[SEP]", "")
    s = " ".join(s.split())
    s = s.replace(" ' s", " 's")
    s = s.replace("[UNK]", "<|unk|>")
    s = s.replace("u . s .", "u.s.")
    s = s.replace("u. s .", "u.s.")
    s = s.replace("[unused2]", "<|unk|>")
    s = s.replace("[unused3]", "<blank>")
    s = s.replace("<blank>,", "<blank> ,")
    return s

def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))
        temp = trainer.get_valid_iterator(subset)
        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        num_samples = max(np.concatenate(trainer.get_valid_iterator(subset).batch_sampler)) +1
        hyps = [[] for i in range(0,num_samples)]
        refs = [[] for i in range(0,num_samples)]



        def decode(toks, escape_unk=False):
            if not cfg.task.use_bert_tokens:
                s = trainer.task.tgt_dict.string(toks.int().cpu(), trainer.task.args.eval_bleu_remove_bpe,
                                                 unk_string=("<|unk|>"))
                if trainer.task.tokenizer:
                    s = trainer.task.tokenizer.decode(s)
            else:
                s = trainer.model.bert_tokenizer.decode(toks)
                # If using bert tokens, we have to map the special tokens back to the normal token
                pseudo2word_dict = json.load(open("pseudo2word_dict.json"))
                s = replace_words(s, pseudo2word_dict)

            return s

        with metrics.aggregate(new_root=True) as agg:
            total_time = 0
            count = 0
            for sample in tqdm(progress):
                if subset == "valid" or subset == "valid_ref":
                    rand_num = np.random.randint(low=0,high=50) # we have 180 K data, which is too many, we want to use 18K instead (change to some very slow number during debugging)
                    if rand_num != 0:
                        continue
                temp_hypo_list = [[]*len(sample)]
                trainer.valid_step(sample)
                processed_sample, _ = trainer._prepare_sample(sample)
                # temp = trainer.task._inference_with_bleu(trainer.task.sequence_generator, processed_sample, trainer.model)
                start_time = time.time()
                gen_out = []
                for i in range(0, processed_sample["nsentences"]):
                    # We independently perform inference for each sample such that we can get an accurate estimate of
                    # the efficiency of different inference algorithms
                    count +=1
                    test_sample = decode(processed_sample["net_input"]["src_tokens"][i].unsqueeze(0)).split()

                    current_sample = {}
                    current_sample["id"] = processed_sample["id"][i]
                    current_sample["nsentences"] = 1
                    current_sample["net_input"] = {}
                    current_sample["net_input"]['src_tokens'] = processed_sample["net_input"]["src_tokens"][i].unsqueeze(0)
                    current_sample["net_input"]['src_lengths'] = processed_sample["net_input"]['src_lengths'][i]
                    current_sample["net_input"]['prev_output_tokens'] = processed_sample["net_input"]['prev_output_tokens'][i].unsqueeze(0)
                    current_sample["target"] = processed_sample["target"][i].unsqueeze(0)
                    current_sample["n_tokens"] = len(current_sample["net_input"]['src_tokens'][0])  # i=8
                    current_gen_out = trainer.task.inference_step(trainer.task.sequence_generator, [trainer.model], current_sample, prefix_tokens=None)
                    gen_out.append(current_gen_out[0])
                end_time = time.time()
                total_time+= end_time-start_time

                def my_remove_adjacent(nums):
                    return [a for a, b in zip(nums, nums[1:] + [not nums[-1]]) if a != b]

                for i in range(len(gen_out)):
                    temp_hypo = decode(gen_out[i][0]["tokens"])
                    if cfg.task.use_transformer:
                        temp_hypo = " ".join(temp_hypo.split()[0:cfg.task.desired_length])
                    try:
                        if cfg.task.no_repeat_ctc:
                            # If we are training with no_repeat ctc loss, we decode it in special way
                            temp_hypo = list(filter(("<blank>").__ne__, temp_hypo.split()))
                            temp_hypo = " ".join(temp_hypo)

                        elif cfg.task.plain_ctc:
                            try:
                                temp_hypo_no_repeat = my_remove_adjacent(temp_hypo.split())
                                temp_hypo_no_repeat = list(filter(("<blank>").__ne__, temp_hypo_no_repeat))
                                temp_hypo = " ".join(temp_hypo_no_repeat)
                            except:
                                print("No blank")
                    except:
                        pass
                    #print(sample)
                    # Remove any potential blank token that still stick in the generated summary
                    temp_hypo = temp_hypo.replace("<blank>", "")
                    temp_hypo = " ".join(temp_hypo.split())
                    temp_ref = decode(
                            utils.strip_pad(processed_sample["target"][i], trainer.task.tgt_dict.pad()),)
                    if cfg.task.length_control == "truncate":
                        temp_hypo = " ".join(temp_hypo.split()[0:cfg.task.desired_length])
                    hyps[sample["id"][i]] = [temp_hypo]
                    refs[sample["id"][i]] =[[temp_ref]]
        # log validation stats
            # print(len(hyps))
            # print("\n \n \n")
        hyps = [hyp for hyp in hyps if len(hyp) != 0]
        if subset == "test":
            with open("giga_ref_summaries.txt") as f:
                refs = f.readlines()
            refs = [[[sentence.rstrip()]] for sentence in refs]
        elif subset == "duc2004":
            # If we are evaluating DUC dataset where we have multiple reference summaries:
            refs = get_duc2004_ref_summary()
        else:
            refs = [ref for ref in refs if len(ref) != 0]
        refs = [ref for ref in refs if len(ref) != 0]
        print("Efficiency per sample:", total_time/(len(hyps)+1))
        # print(refs)
        # print(hyps)
        print("Calculating rouge score...\n")
        if subset =="duc2004":
            has_length_limit = True
            rouge_length = 75
        else:
            has_length_limit = False
            rouge_length = 100
        rouge = Pythonrouge(summary_file_exist=False,
                                summary=hyps, reference=refs,
                                n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                                stemming=True, stopwords=False,
                                word_level=False, length_limit=has_length_limit, length=rouge_length,
                                use_cf=True, cf=95, scoring_formula='average',
                                resampling=True, samples=1000, favor=True, p=0.5)
        rouge_scores = rouge.calc_score()
        save_path = "epoch_%d_step_%d" % (epoch_itr.epoch, epoch_itr.n)
        save_path = os.path.join(cfg.checkpoint.save_dir, save_path)
        save_path = os.path.join(save_path, subset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, "generated_summary.txt"), "w+") as f:
            for item in hyps:
                f.write("%s\n" % item[0])
        with open(os.path.join(save_path,"ref_summary.txt"), "w+") as f:
            for item in refs:
                f.write("%s\n" % item[0][0])

        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        if subset =="duc2004":
            stats["ROUGE-1-R"] = 100*rouge_scores["ROUGE-1-R"]
            stats["ROUGE-2-R"] = 100*rouge_scores["ROUGE-2-R"]
            stats["ROUGE-L-R"] = 100*rouge_scores["ROUGE-L-R"]
            stats["mean_rouge"] = (100*rouge_scores["ROUGE-1-R"] + 100*rouge_scores["ROUGE-2-R"]
                                   + 100*rouge_scores["ROUGE-L-R"])/3
            geo_rouge = np.array([stats["ROUGE-1-R"], stats["ROUGE-2-R"], stats["ROUGE-L-R"]])
        else:
            stats["ROUGE-1-F"] = 100*rouge_scores["ROUGE-1-F"]
            stats["ROUGE-2-F"] = 100*rouge_scores["ROUGE-2-F"]
            stats["ROUGE-L-F"] = 100*rouge_scores["ROUGE-L-F"]
            stats["mean_rouge"] = (100*rouge_scores["ROUGE-1-F"] + 100*rouge_scores["ROUGE-2-F"]
                                   + 100*rouge_scores["ROUGE-L-F"])/3
            geo_rouge = np.array([stats["ROUGE-1-F"], stats["ROUGE-2-F"], stats["ROUGE-L-F"]])
        geo_rouge = geo_rouge.prod() ** (1.0 / len(geo_rouge))
        stats["geo_rouge"] = geo_rouge
        length_list = [len(sentence[0].split()) for sentence in hyps]
        plt.hist(length_list)
        plt.xlabel("summary length")
        plt.ylabel("occurrence")
        plt.savefig(os.path.join(save_path, "summary_length.png"))
        plt.clf()


        stats["ave_length"] = sum(length_list)/len(length_list)
        stats = {subset+"_"+k:v for k, v in stats.items()}
        progress.print(stats, tag=subset, step=trainer.get_num_updates())
        #valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
        if subset == "valid" or subset == "valid_ref" or subset == "duc2004":
            valid_losses.append(geo_rouge)
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
, rouge_score=0) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        if cfg.checkpoint.best_checkpoint_metric == "rouge":
            stats[key] = checkpoint_utils.save_checkpoint.best  # the save_metric is also rouge in our setting.
        else:
            print(stats)
            stats[key] = best_function(
                checkpoint_utils.save_checkpoint.best,
                stats[cfg.checkpoint.best_checkpoint_metric],
            )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
