# CALM

## Overview
Code and data for paper **Keep CALM and Explore: Language Models for Action Generation in Text-based Games** at EMNLP 2020.

* ``games/``: 28 Jericho game files
* ``lm/``: GPT and n-gram CALM models
* ``calm/``: scripts to train and evaluate CALM, and the Clubfloyd dataset
* ``drrn/``: RL agent for gameplay after training CALM


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
The Clubfloyd dataset is crawled from http://www.allthingsjacq.com/interactive_fiction.html. Thanks Jacqueline for hosting this fun cite and granting our use!

The code borrows from [TDQN](https://github.com/microsoft/tdqn) (for the RL part) and [Huggingface Transformers](https://github.com/huggingface/transformers) (for the CALM part). 

For any questions please contact Shunyu Yao `<shunyuyao.cs@gmail.com>`.

