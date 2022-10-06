# CoLo (extractive version)
Code for COLING 2022 paper: *[CoLo: A Contrastive Learning based Re-ranking Framework for One-Stage Summarization]()*
A lightweight re-ranking based summarization system that is **x7** faster than the two-stage system [MatchSum](https://arxiv.org/pdf/2209.14569v1.pdf) while maintaining on-par performance. 

Further increasing the size of the candidate set will result in better performance but will not significantly decrease the speed.

## Dependencies
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.7 +
- [pyrouge](https://github.com/bheinzerling/pyrouge) 0.1.3
	- You should install pyrouge package first to reproduce our results. Instruction for installing pyrouge can be found in this [repo](https://github.com/ChenxinAn-fdu/CGSum)
- [rouge](https://github.com/pltrdy/rouge) 1.0.0
	- Used in  the validation phase.
- [transformers](https://github.com/huggingface/transformers) 4.10 + (for extractive verison)
- [transformers](https://github.com/huggingface/transformers) 4.20 + (for abstractive verison)


All code only supports running on Linux.

## Data

We have already processed CNN/DailyMail dataset (document-max-length = 512), 
you can download it through [this link](https://drive.google.com/file/d/1vCpTPyZwDFIcQ4yZX4YXLdqjugmxfmka/view?usp=sharing), and move it to `./datasets/CNNDM/`.
You can also download the raw data from [here](https://drive.google.com/file/d/1YXPJYcu5WRorfiFRGw70brGs1RKEQfHk/view?usp=sharing), but remember to run our preprocess script  `preprocess/ext_label_and_tokenize.py` for pre-process. The raw dataset must take the format of `jsonl`.  If you want to try other dataset or use the document-max-length = 1024, please run the preprocess script before training.


## Train

We use 8 RTX 3090 24G GPUs to train our model. If you want use longer input, you can set a smaller batch size and set the `accum_count>1`.        

You can choose BART-encoder/Roberta/BERT as the encoder of **CoLo** (extractive). BART and roberta share the same tokenizer, and remember to re-preprocess the raw data with BERT-tokenizer if you want to use BERT as base model.
We select the encoder of BART for its longer max-position (1024) 
To **train** a one-stage re-ranking summarization model on CNNDM, you can use the `run_cnndm.py`:

```
#If you do not have a warmed-up checkpoint, you should use --warmup True to train the extractive model with BCELoss 
python run_cnndm.py --mode train --gpus 0,1,2,3 --batch_size 64 --warmup True
```
the warmed-up checkpoint will be saved to `./warmed_up/CNNDM/pretrain.ext.pt` by default. We also prepare a [warmed-up checkpoint](https://drive.google.com/file/d/11rAC5ghms7NLmdJlRBRbrE2JDeFcpedV/view?usp=sharing) to help skip the warmup stage.
```
#If you already have a warmed-up checkpoint:
mv path/to/the/checkpoint.pt ./warmed_up/CNNDM/pretrain.ext.pt
python run_cnndm.py --mode train --gpus 0,1,2,3 --batch_size 64 --warmup False
```
After completing the training process, several best checkpoints are stored in a folder named after the training start time, for example, `./checkpoints/CNNDM/2022-05-22-09-24-51`. 

## Test
You can run the following command to get the results on test set:

```
python train.py --mode test --save_path checkpoints/CNNDM/2022-05-22-09-24-51 --gpus 0
```
This will produce the generated results at: `results/CNNDM/2022-05-22-09-24-51/filename.jsonl`


## Evaluation
This is an example to evaluate the generated results with pyrouge
```
python ../evaluation/eval_with_pyrouge.py --sys_path results/CNNDM/2022-05-22-09-24-51/filename.jsonl
```

## Citing
Please cite our work if you find this paper or codes useful.
```
@article{an2022colo,
  title={COLO: A Contrastive Learning based Re-ranking Framework for One-Stage Summarization},
  author={An, Chenxin and Zhong, Ming and Wu, Zhiyong and Zhu, Qin and Huang, Xuanjing and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2209.14569},
  year={2022}
}
```


