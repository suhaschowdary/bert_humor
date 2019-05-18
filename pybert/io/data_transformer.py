#encoding:utf-8
import random
import operator
import pandas as pd
from tqdm import tqdm
from collections import Counter
from ..utils.utils import text_write
from ..utils.utils import pkl_write

class DataTransformer(object):
    def __init__(self,
                 logger,
                 seed,
                 add_unk = True
                 ):
        self.seed          = seed
        self.logger        = logger
        self.item2idx = {}
        self.idx2item = []
        if add_unk:
            self.add_item('<unk>')

    def add_item(self,item):
        '''
        :param item:
        :return:
        '''
        item = item.encode('UTF-8')
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1

    def get_idx_for_item(self,item):
        '''
        :param item:
        :return:
        '''
        item = item.encode('UTF-8')
        if item in self.item2idx:
            return self.item2idx[item]
        else:
            return 0

    def get_item_for_index(self, idx):
        '''
        id，tokens
        :param idx:
        :return:
        '''
        return self.idx2item[idx].decode('UTF-8')

    def get_items(self):
        '''
        items
        :return:
        '''
        items = []
        for item in self.idx2item:
            items.append(item.decode('UTF-8'))

    def split_sent(self,line):
        res = line.strip('\n').split()
        return res

    def train_val_split(self,X, y,valid_size,
                        stratify=False,
                        shuffle=True,
                        save = True,
                        train_path = None,
                        valid_path = None):
        '''
        # train and validate
        :return:
        '''
        self.logger.info('train val split')
        if stratify:
            num_classes = len(list(set(y)))
            train, valid = [], []
            bucket = [[] for _ in range(num_classes)]
            for data_x, data_y in tqdm(zip(X, y), desc='bucket'):
                bucket[int(data_y)].append((data_x, data_y))
            del X, y
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                test_size = int(N * valid_size)
                if shuffle:
                    random.seed(self.seed)
                    random.shuffle(bt)
                valid.extend(bt[:test_size])
                train.extend(bt[test_size:])
            # train
            if shuffle:
                random.seed(self.seed)
                random.shuffle(train)
        else:
            data = []
            for data_x, data_y in tqdm(zip(X, y), desc='Merge'):
                data.append((data_x, data_y))
            del X, y
            N = len(data)
            test_size = int(N * valid_size)
            if shuffle:
                random.seed(self.seed)
                random.shuffle(data)
            valid = data[:test_size]
            train = data[test_size:]
            # train
            if shuffle:
                random.seed(self.seed)
                random.shuffle(train)
        if save:
            text_write(filename=train_path, data=train)
            text_write(filename=valid_path, data=valid)
        return train, valid

    def build_vocab(self,data,min_freq,max_features,save,vocab_path):
        '''
        :param data:
        :param min_freq:
        :param max_features:
        :param save:
        :param vocab_path:
        :return:
        '''
        count = Counter()
        self.logger.info('Building word vocab')
        for i,line in enumerate(data):
            words = self.split_sent(line)
            count.update(words)
        count = {k: v for k, v in count.items()}
        count = sorted(count.items(), key=operator.itemgetter(1))
        all_words = [w[0] for w in count if w[1] >= min_freq]
        if max_features:
            all_words = all_words[:max_features]

        self.logger.info('vocab_size is %d' % len(all_words))
        for word in all_words:
            self.add_item(item = word)
        if save:
            pkl_write(data = self.item2idx,filename = vocab_path)

    def read_data(self,raw_data_path,preprocessor = None,is_train=True):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        targets, sentences = [], []
        data = pd.read_csv(raw_data_path)
        for row in tqdm(data.values):
            if is_train:
                target = row[1]
            else:
                target = row[1]
            sentence = str(row[0])
            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                targets.append(target)
                sentences.append(sentence)
        return targets,sentences
