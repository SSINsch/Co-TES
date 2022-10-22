import os
import os.path
import numpy as np
import pickle
import logging

import torch
import torch.utils.data as data
from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer
from torchtext.data import get_tokenizer

from utils import noisify

logger = logging.getLogger(__name__)


def regroup_dataset(labels):
    """
    categories = dataset.target_names
    labels = [(dataset.target_names.index(cat), cat) for cat in categories]
    [(0, 'alt.atheism'), (1, 'comp.graphics'), (2, 'comp.os.ms-windows.misc'), (3, 'comp.sys.ibm.pc.hardware'), (4, 'comp.sys.mac.hardware'), (5, 'comp.windows.x'), (6, 'misc.forsale'), (7, 'rec.autos'), (8, 'rec.motorcycles'), (9, 'rec.sport.baseball'), (10, 'rec.sport.hockey'), (11, 'sci.crypt'), (12, 'sci.electronics'), (13, 'sci.med'), (14, 'sci.space'), (15, 'soc.religion.christian'), (16, 'talk.politics.guns'), (17, 'talk.politics.mideast'), (18, 'talk.politics.misc'), (19, 'talk.religion.misc')]
    """
    batch_y = labels.copy()
    for i, label in enumerate(labels):
        if label in [0]:
            batch_y[i] = 0
        if label in [1, 2, 3, 4, 5, ]:
            batch_y[i] = 1
        if label in [6]:
            batch_y[i] = 2
        if label in [7, 8, 9, 10]:
            batch_y[i] = 3
        if label in [11, 12, 13, 14]:
            batch_y[i] = 4
        if label in [15]:
            batch_y[i] = 5
        if label in [16, 17, 18, 19]:
            batch_y[i] = 6

    logger.info(f'regrouped label {batch_y.shape}')

    return batch_y


class NewsGroupsOriginal(data.TensorDataset):
    """
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root='./data', train=True, transform=None, target_transform=None,
                 noise_type=None, noise_rate=0.2, random_state=0):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.noise_type = noise_type
        self.dataset = 'news'
        self.weights_matrix, data, labels = pickle.load(open(os.path.join(self.root, "news.pkl"), "rb"),
                                                        encoding='iso-8859-1')
        labels = regroup_dataset(labels)

        length = labels.shape[0]
        self.num_classes = len(set(labels))

        if self.train:
            self.train_data = torch.from_numpy(data[:int(length * 0.70)])
            self.train_labels = torch.from_numpy(labels[:int(length * 0.70)]).long()

            # noisify train data
            if noise_type is not None:
                self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset,
                                                                          nb_classes=self.num_classes,
                                                                          train_labels=self.train_labels,
                                                                          noise_type=noise_type, noise_rate=noise_rate,
                                                                          random_state=random_state)
                self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                _train_labels = [i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels) == np.transpose(_train_labels)
                logger.info(f'label precision: {1 - self.actual_noise_rate}')
        else:
            self.test_data = torch.from_numpy(data[int(length * 0.70):])
            self.test_labels = torch.from_numpy(labels[int(length * 0.70):])

    def __getitem__(self, index):
        if self.train:
            if self.noise_type is not None:
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        mask = 0

        return img, mask, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class NewsGroupsUpdate(data.TensorDataset):
    def __init__(self, root='./data', train=True, noise_type=None, noise_rate=0.2, random_state=0, max_len=512):
        super(NewsGroupsUpdate).__init__()
        self.root = os.path.expanduser(root)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        self.basic_tokenizer = get_tokenizer('basic_english')
        self.max_len = max_len
        self.train = train  # training set or test set
        self.noise_type = noise_type
        self.dataset = 'news'

        # vocab weight matrix (20000, 300) : (20000개 단어 각각 300dim vector)
        self.vocab_weights_matrix = pickle.load(open(os.path.join(self.root, "news_vocab_glove_weight.pk"), "rb"))
        # vocab_itos = ['<unk>', '<pad>', ...]
        self.vocab_itos = pickle.load(open(os.path.join(self.root, "news_vocab_list.pk"), "rb"))
        # vocab_stoi = {'<unk>': 0, '<pad>': 1, ...}
        self.vocab_stoi = {k: v for v, k in enumerate(self.vocab_itos)}

        # Todo.__getitem__에서 사용하는 encoding 결과들을 미리 저장해두고 불러와서 속도를 높이자.
        # if 문으로 처리해서 pk이 있으면 로드하고 아니면 tokenizer 선언 후 tokenize하도록 변경하면 될듯.

        if self.train is True:
            raw_data = fetch_20newsgroups(subset='train')
        else:
            raw_data = fetch_20newsgroups(subset='test')
        self.sentences = np.asarray(raw_data.data)
        labels = raw_data.target
        labels = regroup_dataset(labels)
        self.num_classes = len(set(labels))
        self.labels = torch.from_numpy(labels).long()

        # noisify train data label
        if (self.noise_type is not None) and (self.train is True):
            self.labels = np.asarray([[self.labels[i]] for i in range(len(self.labels))])
            self.noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset,
                                                                nb_classes=self.num_classes,
                                                                train_labels=self.labels,
                                                                noise_type=noise_type,
                                                                noise_rate=noise_rate,
                                                                random_state=random_state)
            self.noisy_labels = [i[0] for i in self.noisy_labels]
            _clean_labels = [i[0] for i in self.labels]
            self.noise_or_not = np.transpose(self.noisy_labels) == np.transpose(_clean_labels)
            logger.info(f'label precision: {1 - self.actual_noise_rate}')

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        text = self.sentences[index]
        bert_encoding = self.bert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        basic_encoding = []
        basic_tokenized_text = self.basic_tokenizer(text)
        for tok in basic_tokenized_text:
            try:
                basic_encoding.append(self.vocab_stoi[tok])
            except KeyError as ke:
                basic_encoding.append(self.vocab_stoi['<unk>'])

        if len(basic_encoding) > self.max_len:
            basic_encoding = np.asarray(basic_encoding[:self.max_len])
        else:
            z = np.zeros(self.max_len - len(basic_encoding)).astype(int)
            basic_encoding = np.asarray(basic_encoding)
            basic_encoding = np.append(z, basic_encoding)
        basic_encoding = torch.from_numpy(basic_encoding)

        if (self.noise_type is not None) and (self.train is True):
            target = self.noisy_labels[index]
        else:
            target = self.labels[index]

        input_ids = bert_encoding['input_ids'].flatten()
        att_masks = bert_encoding['attention_mask'].flatten()

        return input_ids, att_masks, basic_encoding, target, index


if __name__ == '__main__':
    test_dataset = NewsGroupsUpdate(root='../data')

