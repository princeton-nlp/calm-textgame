# Contextual Action Language Model (CALM) and the ClubFloyd Dataset

Code and data for paper [Keep CALM and Explore: Language Models for Action Generation in Text-based Games](https://arxiv.org/pdf/2010.02903.pdf) at EMNLP 2020.

## Overview
Our **ClubFloyd dataset** (`calm/lm_data.zip`) crawls from [the ClubFloyd website](http://www.allthingsjacq.com/interactive_fiction.html) 426 human gameplay transcripts, which conver 590 text-based games of diverse genres and styles, and 223,527 context-action pairs of format `[CLS] observation [SEP] action [SEP] next observation [SEP] next action [SEP]`. We use `[CLS] observation [SEP] action [SEP] next observation [SEP]` to train language models (n-gram, GPT-2) to predict `next action [SEP]`, and show this action generation ability generalizes to **unseen games** and supports gameplay when combined with reinforcement learning. 

##  Getting Started
- Clone repo and install dependencies:

```bash
pip install torch==1.4 transformers==2.5.1 jericho fasttext wandb importlib_metadata
git clone https://github.com/princeton-nlp/calm && cd calm
ln -s ../lm calm && ln -s ../lm drrn
```

(If the pip installation fails for fasttext, try the build steps here: https://github.com/facebookresearch/fastText#building-fasttext-for-python)

- Train CALM:
```bash
cd calm
unzip lm_data.zip
python train.py
```

Trained model weights can be downloaded [here](https://drive.google.com/file/d/1PBAXq4LW9pdVdLFyF_donwCV46wBX1zD/view?usp=sharing) for both GPT-2 and n-gram models. 

- Then train DRRN using the trained CALM:
```bash
cd ../drrn
python train.py --rom_path ../games/${GAME} --lm_path ${PATH_TO_CALM} --lm_type ${gpt_or_ngram}
```

## Citation
```
@inproceedings{yao2020calm,
    title={Keep CALM and Explore: Language Models for Action Generation in Text-based Games},
    author={Yao, Shunyu and Rao, Rohan and Hausknecht, Matthew and Narasimhan, Karthik},
    booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
    year={2020}
}
```
## Acknowledgements
Thanks Jacqueline for hosting the wonderful ClubFloyd webcite and granting our use!

The code borrows from [TDQN](https://github.com/microsoft/tdqn) (for the RL part) and [Huggingface Transformers](https://github.com/huggingface/transformers) (for the CALM part). 

For any questions please contact Shunyu Yao `<shunyuyao.cs@gmail.com>`.

