import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from typing import Tuple, List
import logging
from utils import read_jsonl_data, prepare_sequence

logger = logging.getLogger(__name__)


class DevignDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, tok2idx=None):
        self.data = read_jsonl_data(data_path)
        self.func = [x['func'] for x in self.data]
        self.target = [x['target'] for x in self.data]
        self.tokenizer = tokenizer
        self.tok2idx = tok2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x_func = self.func[index]
        y_vuln_target = self.target[index]
        x_func_tokenized = self.tokenizer.tokenize(x_func)

        # get index of words(token) and characters(of each word)
        x_idx_tokens_item = prepare_sequence(x_func_tokenized, self.tok2idx)

        return x_func_tokenized, x_idx_tokens_item, y_vuln_target, index


class BasicCollator:
    def __init__(self, block_size: int):
        self.block_size = block_size

    def __call__(self, data: Tuple[List, List]):
        # do something with batch and self.params
        data.sort(key=lambda x: len(x[0]), reverse=True)

        x_func_tokenized, x_idx_tokens_item, y_vuln_target, index = zip(*data)

        # label -> Torch
        y_vuln_target_torched = torch.tensor(y_vuln_target)

        # index data -> Torch
        # get max length of input
        x_lens = [len(x) for x in x_idx_tokens_item]
        max_x_len = max(x_lens)
        if max_x_len < self.block_size:
            _block_size = max_x_len
        else:
            _block_size = self.block_size

        # padding & slicing
        blocked_x_idx_tokens_item = []
        for i, x in enumerate(x_idx_tokens_item):
            if len(x) < _block_size:
                padding_list = [0] * (_block_size - len(x))
                blocked_x_idx_tokens_item.append(x + padding_list)
            else:
                blocked_x_idx_tokens_item.append(x[:_block_size])

        padded_blocked_x_idx_tokens_matrix = torch.tensor(blocked_x_idx_tokens_item)
        index = np.asarray(index)
        index = torch.from_numpy(index)
        # index = torch.tensor(index)

        return x_func_tokenized, padded_blocked_x_idx_tokens_matrix, y_vuln_target_torched, index


def get_loader(data_path: str,
               tokenizer,
               block_size: int,
               tok2idx=None,
               collate_method: str = 'collate_basic',
               batch_size: int = 256,
               shuffle: bool = True):
    collate = None
    if collate_method == 'collate_basic':
        # collate = collate_basic
        collate = BasicCollator(block_size=block_size)
    else:
        logger.warning(f'Unknown collate method: {collate_method}')

    logger.debug(f'load dataset with {data_path}')
    dataset = DevignDataset(data_path, tokenizer, tok2idx=tok2idx)

    logger.debug('load dataloader')
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=collate)
    return data_loader, len(dataset)
