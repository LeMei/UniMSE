# UniMSE

Paper: UniMSE: Towards Unified Multimodal Sentiment Analysis and Emotion Recognition EMNLP(2022)

Arxiv: https://arxiv.org/pdf/2211.11256.pdf

Abstract:

Multimodal sentiment analysis (MSA) and emotion recognition in conversation (ERC) are key research topics for computers to understand human behaviors. From a psychological perspective, emotions are the expression of affect or feelings during a short period, while sentiments are formed and held for a longer period. However, most existing works study sentiment and emotion separately and do not fully exploit the complementary knowledge behind the two. In this paper, we propose a multimodal sentiment knowledge-sharing framework (UniMSE) that unifies MSA and ERC tasks from features, labels, and models. We perform modality fusion at the syntactic and semantic levels and introduce contrastive learning between modalities and samples to better capture the difference and consistency between sentiments and emotions. Experiments on four public benchmark datasets, MOSI, MOSEI, MELD, and IEMOCAP, demonstrate the effectiveness of the proposed method and achieve consistent improvements compared with the state-of-art methods.

Architecture:

![image](https://user-images.githubusercontent.com/22788642/196078764-4313a0ac-9727-4692-9836-99f667007c93.png)


Code: we will open the source codes in the future, please wait patiently.

Features: 

链接:https://pan.baidu.com/s/190tw6g0bPkiOPu5xPpvDTQ  
密码:xyg5

链接:https://pan.baidu.com/s/17n_Hi2Tv0a7qsPoOp6qLKw 
密码:rms6

链接:https://pan.baidu.com/s/1VOgH_BW08TxVEQto8fUvfw
密码:txvm


Text Information链接:https://pan.baidu.com/s/1wBLZnE4-FACV80iPQgC4og  密码:298s




## Usage

1. datasets

MOS -- MOSI + MOSEI
MOSELD -- MOSI+MOSEI+MELD
MOSELDMP -- MOSI+MOSEI+MELD+IEMOCAP



2. Environment

torch                        1.7.0+pai

torchvision                  0.8.0

tensorboardX                 2.5

tensorflow-estimator         1.15.1

tensorflow-gpu               1.15.0

transformers                 4.12.5

Running

First, generate Universal Label for each dataset -> Simcse folder, versation 3

Second, generate Train dataset based on single dataset MOSI, MOSEI, IEMOCAP，MELD -> preprocess.py & create_dataset.py

Third， setting the dataset folder for dataloader. ->main.py


