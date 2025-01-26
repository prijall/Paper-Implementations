# BERT 
- BERT stands for Bidirectional Encoder Representations from Transformers. 

## Abstract

- BERT is a new language representation model. BERT is different from other models in the sense that it pretrain deep bidirectional representations by jointly conditioning  on both left and right context in all layers.

- It can be fine-tuned with additional output layer for varieties of tasks such as question answering.

## Architecture:

There are two steps in bert, they are: **Pre-training** and **fine-tuning**.

- **Pre-training:** it is the process where we train our model with unlabelled data
over different pre-training task(teaches bert foundational langauge understanding).

- **Fine-tuning:** after pre-training is done, the model is initialized with pre-trained
parameters and these parameters are fine-tuned using labelled data from downstream task(adapt
bert to solve real-world problems).

BERT's model Architecture is multi-layer bidirectional Transformer encoder based on original 
implementaion on the paper **attention is all you need**. There are two model sizes:

BERT<sub>BASE</sub>(number of layers(L)=12, Hidden Size(H)=768, number of self-attention heads(A)=12, Total parameters=110M).
BERT<sub>LARGE</sub>(L=24, H=1024, A=16, Total parameters=340M)

![alt text](Photo/BERT_ARCHITECTURE.png)