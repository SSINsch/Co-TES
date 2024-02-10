import os
import os.path
import numpy as np
import torch
import pickle
import logging
import torch.utils.data as data

from utils import noisify

logger = logging.getLogger(__name__)


class Yahoo(data.TensorDataset):
    def __init__(self,
                 root='./data', train=True, transform=None, target_transform=None,
                 noise_type=None, noise_rate=0.2, random_state=0):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.noise_type = noise_type
        self.dataset = 'yahoo'
        self.num_classes = 10

        if self.train:
            self.train_data, self.train_labels = pickle.load(open(os.path.join(self.root, "yahoo-train.pkl"), "rb"), encoding='iso-8859-1')
            self.train_labels = torch.from_numpy(self.train_labels).long()

            # noisify train data
            if noise_type is not None:
                self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset,
                                                                          nb_classes=self.num_classes,
                                                                          train_labels=self.train_labels,
                                                                          noise_type=noise_type,
                                                                          noise_rate=noise_rate,
                                                                          random_state=random_state)
                self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
                _train_labels = [i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels) == np.transpose(_train_labels)
                logger.info(f'label precision: {1 - self.actual_noise_rate}')

        else:
            self.test_data, self.test_labels = pickle.load(open(os.path.join(self.root, "yahoo-test.pkl"), "rb"), encoding='iso-8859-1')
            self.test_labels = torch.from_numpy(self.test_labels).long()

    def __getitem__(self, index):
        if self.train:
            if self.noise_type is not None:
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
