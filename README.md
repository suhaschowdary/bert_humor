# Humor detector using Bert
In this repo, we show how to setup and run a humor detector model using BERT.


## Setup environment

This model requires PyTorch 1.0 or later. We recommend running the code in a virtual environment with Python 3.6:
```
virtualenv -p python3.6 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

deactivate `source deactivate`


## Steps to run the model

**GPU with minimum 16GB RAM is required to run the model ** 

1. Clone this repo and setup the virtual environment.
2. Downlaod the [pretrained BERT model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and store it to `pybert/model/pretrain/`.
3. As google provides models pretrained on tensorflow, run `convert_tf_checkpoint_to_pytorch.py` to make the pretrained weights compatible with Pytorch.
4. Run `train_bert_humor.py` to train the humor detector model.
5. You can also use `inference.py` to evaluate the models with finetuned models.

## Run model on google colab
You can also run the model directly using `bert_humor.ipynb` notebook using google colab. But you should first downlaod the [pretrained BERT model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and save it to `pybert/model/pretrain/` before running the model. 

## Data preparation

We used 2 datasets to train the model which are availabe in `data` folder.
1. Sarcasm data used in this repo is directly collected from [News Headlines Dataset For Sarcasm Detection](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection).
2. humor data is prepared from a couple of sources like [Short jokes dataset](https://www.kaggle.com/abhinavmoudgil95/short-jokes) and
[kaggles all the news data](https://www.kaggle.com/snapcrack/all-the-news).

You are free to use any dataset as long as it is in required format.

## Fine tuning
Fine tuning can be done according to the model requirements. One can change the parameters in `pybert/config/basic_config.py` to tune the model. 
Some hyperparameters which are used to train this model and can be explored further are:
```
max_seq_len = 256
do_lower_case = True
batch_size = 20
epochs = 3
warmup_proportion = 0.1
gradient_accumulation_steps = 1
learning_rate = 2e-5
weight_decay = 1e-5
```

## Results

The results on the 2 datasets are as follows:

| Training dataset              | Test dataset               | F1    | accuracy |
|:-----------------------------:|:--------------------------:|:-----:|:--------:|
| News headlines training data  | News headline test data    | 0.97  | 0.99     |
| jokes+articles trainning data | jokes+articles test data   | 0.91  | 0.94     | 
| jokes+articles trainning data | complete new headlines data| 0.83  | 0.85     |


## References
1. [BERT paper](https://arxiv.org/abs/1810.04805)
2. Hugging face's pretrained pytorch models: [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
3. [pytorch multi-label classification](https://github.com/lonePatient/Bert-Multi-Label-Text-Classification) for jigsaw toxic comment classification




