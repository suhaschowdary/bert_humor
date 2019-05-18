# Humor detector using Bert


#### Objective
To built a humor detector. In simple words parse a sentece to a model, we have to identify whether it is humorous or not!!


## Setup

This model requires PyTorch 1.0 or later. We recommend running the code in a virtual environment with Python 3.6:
```
virtualenv -p python3.6 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

deactivate `source deactivate`


#### Metrics 
The most elegant way or most proficient way is to identify the humor content in the sentence. Maybe if we can quantify the humor 
like the sentence is 60% humorous would be super cool. However we need proper language experts to label these senteces and opinions 
may vary! So let's consider the problem to be a classification problem. So lets consider accuracy, precision, recall and F1 scores as 
evaluation metrics.

#### Data collection
Now first task is to collect the data. We do not really have proper training data. However after reading this paper
([Humor recognition using deep learning](https://www.aclweb.org/anthology/N18-2018)) provided in the exercise
and from the datasets sent in the exercise , I have got some fair idea about data preparation.
So I collected data from 16000 one liners, [short jokes dataset](https://www.kaggle.com/abhinavmoudgil95/short-jokes) and 
[News Headlines Dataset For Sarcasm Detection](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection). 
Also to make the negative sample distribution proportional, I collected new headlines from [WMT](http://www.statmt.org/wmt16/translation-task.html) 
as suggested in paper. I want to collect more data and see how it works. However my first task is to build a simple baseline model and
make it work. I would like to improve on it later.

#### Building baseline model

1. Prepare the data set for humor.
2. Pretrain the model on `Bert`
3. Use semi-supervised learning and transfer learning to train `Bert` on humor dataset(thanks to the article sent on using [semi-supervised learning and transfer learning(https://towardsdatascience.com/a-technique-for-building-nlp-classifiers-efficiently-with-transfer-learning-and-weak-supervision-a8e2f21ca9c8)).
4. Fine tune the model on our huor detector task and evaluate the model.

#### Tasks

- [x] Prepare dataset
- [x] Get accustomed with google colab - Trained simple bert model on some random classification task.
- [x] Build a crude model using Bert which can be served as a baseline model.
- [x] Evaluate the baseline model and use one shot learning to see where it goes.
- [ ] Improve the baseline model and explore semi-supervised learning!

