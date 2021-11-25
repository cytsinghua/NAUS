# NAUS
This repo contains the code to replicate experiments in [Learning Non-Autoregressive Models from Search for Unsupervised Sentence Summarization](https://openreview.net/forum?id=UNzc8gReN7m).

Prepare
=======

### Python Version and Requirements
The script is developed based on [fairseq](https://github.com/pytorch/fairseq) and is tested with python version 3.8.

Our original implementation is done with Anaconda, and we include the commands to setup its enviroment in conda_commands.txt

We also offer a python virtual enviroment approach to set up the enviroment, but some packages (e.g., CUDA) needs to be mannually configured. 
Use the following command to install the required packages in a python virtual enviroment: 

```
pip install -r requirements.txt
```

After setting up the enviroment, our script can be installed by 

```
pip install -e .
```

### Download Data
Download training [Gigaword for headline generation](https://github.com/harvardnlp/sent-summary) data.

### Obtain Search Result
In our approach, the model is trained from the pseudo-summaries generated by [search](https://github.com/raphael-sch/HC_Sentence_Summarization).
After getting the search output (assuming you want to train NAUS with pseudo-summaries of length **10**), create a folder **gigaword_10** and place the train, validation and test data into this folder. 
Specifically, you should name your train/valid/test input and output as train/valid/test.article and train/valid/test.summary respectively. 


### Preprocess the data

Assume your folder is **gigaword_10**, use the following command to preprocess the data
```
TEXT=gigaword_10
python3 fairseq_cli/preprocess.py --source-lang article --target-lang summary --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/$TEXT --workers 40 --joined-dictionary
```


Model Training
=============
We first give a demonstration on training NAUS on 10 words summaries, and set the desired summary length to be 10.
To do this, we declare some variables in the terminal:

```
data_source=data-bin/gigaword_10
arch=nat_encoder_only_customized_ctc
length_control=beam_search_length_control
desired_length=10
beam_size=6
k=20
plain_ctc=False
valid_subset=test,valid
drop_out=0.1
max_token=4096
max_update=200000
use_bert_tokens=False
```

Then run the training script:

	CUDA_VISIBLE_DEVICES=0 python train.py $data_source --source-lang article --target-lang summary --save-dir giga_${arch}_${length_control}_${desired_length}_plain_ctc_${plain_ctc}_use_bert_tokens_${use_bert_tokens}_beam_size_${beam_size}_k_${k}_dropout_${drop_out}_checkpoints --eval-tokenized-rouge True --keep-interval-updates 5 --save-interval-updates 5000 --validate-interval-updates 5000 --maximize-best-checkpoint-metric --eval-rouge-remove-bpe True --eval-rouge-print-samples True --best-checkpoint-metric loss --log-format simple --log-interval 100 --eval-rouge True --keep-last-epochs 5 --keep-best-checkpoints 5 --fixed-validation-seed 7 --ddp-backend=no_c10d --share-all-embeddings --decoder-learned-pos --encoder-learned-pos --optimizer adam --adam-betas "(0.9,0.98)" --lr 0.0005 --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 --warmup-updates 10000 --warmup-init-lr 1e-07 --apply-bert-init --weight-decay 0.01 --fp16 --clip-norm 2.0 --max-update $max_update --task translation_lev --criterion nat_loss --arch $arch --noise full_mask --src-upsample-scale 1 --use-ctc-decoder --ctc-beam-size 1 --label-smoothing 0.1 --activation-fn gelu --dropout $drop_out --max-tokens $max_token --eval-bleu-remove-bpe --valid-subset $valid_subset --plain_ctc $plain_ctc --length_control $length_control --desired_length $desired_length --k $k --use_bert_tokens $use_bert_tokens --beam_size $beam_size --use_length_ratio False --force_length False

To explain: 

**length_control** refers to the method to control the output length, it can be chosen from "no_control", "truncate" and "beam_search_length_control";

**desired_length** refers to the desired length of the output summary. It will be ignored if length control is set to no_control. 

**beam_size** refers to the beam search of the beam search component in the length control;

**k** is the number of tokens we consider at each time slot during the beam search;

**plain_ctc** determines whether to use plain CTC decoding;

**valid_subset** is the subset of the validation and test dataset, "valid" is always required to perform validation;

**max_token** controls the max token in each batch;

**use_bert_tokens** is not usefult for our model, always set to False.


To run the training script and perform inference with our length-control algorithm with a specified length (length-transfer), set the force_length to be true:

	CUDA_VISIBLE_DEVICES=0 python train.py $data_source --source-lang article --target-lang summary --save-dir giga_${arch}_${length_control}_${desired_length}_plain_ctc_${plain_ctc}_use_bert_tokens_${use_bert_tokens}_beam_size_${beam_size}_k_${k}_dropout_${drop_out}_checkpoints --eval-tokenized-rouge True --keep-interval-updates 5 --save-interval-updates 5000 --validate-interval-updates 5000 --maximize-best-checkpoint-metric --eval-rouge-remove-bpe True --eval-rouge-print-samples True --best-checkpoint-metric loss --log-format simple --log-interval 100 --eval-rouge True --keep-last-epochs 5 --keep-best-checkpoints 5 --fixed-validation-seed 7 --ddp-backend=no_c10d --share-all-embeddings --decoder-learned-pos --encoder-learned-pos --optimizer adam --adam-betas "(0.9,0.98)" --lr 0.0005 --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 --warmup-updates 10000 --warmup-init-lr 1e-07 --apply-bert-init --weight-decay 0.01 --fp16 --clip-norm 2.0 --max-update $max_update --task translation_lev --criterion nat_loss --arch $arch --noise full_mask --src-upsample-scale 1 --use-ctc-decoder --ctc-beam-size 1 --label-smoothing 0.1 --activation-fn gelu --dropout $drop_out --max-tokens $max_token --eval-bleu-remove-bpe --valid-subset $valid_subset --plain_ctc $plain_ctc --length_control $length_control --desired_length $desired_length --k $k --use_bert_tokens $use_bert_tokens --beam_size $beam_size --use_length_ratio False --force_length True


To run the training script and perform inference with our length-control algorithm with a specified length ratio (length-transfer), set the use_length_ratio to be true:

	CUDA_VISIBLE_DEVICES=0 python train.py $data_source --source-lang article --target-lang summary --save-dir giga_${arch}_${length_control}_${desired_length}_plain_ctc_${plain_ctc}_use_bert_tokens_${use_bert_tokens}_beam_size_${beam_size}_k_${k}_dropout_${drop_out}_checkpoints --eval-tokenized-rouge True --keep-interval-updates 5 --save-interval-updates 5000 --validate-interval-updates 5000 --maximize-best-checkpoint-metric --eval-rouge-remove-bpe True --eval-rouge-print-samples True --best-checkpoint-metric loss --log-format simple --log-interval 100 --eval-rouge True --keep-last-epochs 5 --keep-best-checkpoints 5 --fixed-validation-seed 7 --ddp-backend=no_c10d --share-all-embeddings --decoder-learned-pos --encoder-learned-pos --optimizer adam --adam-betas "(0.9,0.98)" --lr 0.0005 --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 --warmup-updates 10000 --warmup-init-lr 1e-07 --apply-bert-init --weight-decay 0.01 --fp16 --clip-norm 2.0 --max-update $max_update --task translation_lev --criterion nat_loss --arch $arch --noise full_mask --src-upsample-scale 1 --use-ctc-decoder --ctc-beam-size 1 --label-smoothing 0.1 --activation-fn gelu --dropout $drop_out --max-tokens $max_token --eval-bleu-remove-bpe --valid-subset $valid_subset --plain_ctc $plain_ctc --length_control $length_control --desired_length $desired_length --k $k --use_bert_tokens $use_bert_tokens --beam_size $beam_size --use_length_ratio True --force_length True

Notice, desired length in this case become the desired length ratio. For example, setting the desired length to be 50 will force NAUS to generate summary whose length is 50% of that of its input. 

