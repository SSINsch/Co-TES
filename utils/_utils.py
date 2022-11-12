import os
import argparse
import torch.nn.functional as F
import logging
from transformers import BertTokenizer
from parser_tokenizers import CodeTokenizer
import json
import collections
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, help='now only support news dataset', default='news')
    parser.add_argument('--model1', type=str, help='first model type')
    # Todo. 모델 옵션 추가
    parser.add_argument('--lstm_opt1', type=int, help='first lstm model option', default='300')
    parser.add_argument('--lstm_opt2', type=int, help='second lstm model option', default='300')
    # parser.add_argument('--cnn_opt1', nargs='+', help='first cnn model option', default='3 4 5')
    parser.add_argument('--fcn_opt1', type=int, help='first fcn model option', default='300')
    parser.add_argument('--fcn_opt2', type=int, help='second ccn model option', default='300')
    parser.add_argument('--model2', type=str, help='second model type')
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
    parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='logs/')
    parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='symmetric')
    parser.add_argument('--num_gradual', type=int, default=10,
                        help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')

    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--num_workers', type=int, default=1, help='how many subprocesses to use for data loading')
    parser.add_argument('--epoch_decay_start', type=int, default=80)
    parser.add_argument('--model_type', type=str, help='[coteaching, coteaching_plus]', default='coteaching_plus')
    parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')

    args = parser.parse_args()

    if args.num_gradual > args.n_epoch:
        logging.warning(f'(num_gradual) should be greater than (n_epoch) : {args.num_gradual}, {args.n_epoch}')
        raise ValueError(f'(num_gradual) should be greater than (n_epoch) : {args.num_gradual}, {args.n_epoch}')

    log_m = f'[Args] (seed: {args.seed}), (model1: {args.model1}), '
    log_m += f'(model2: {args.model2}), (noise_rate: {args.noise_rate})'
    logger.debug(log_m)

    return args


def accuracy_top_k(logit, target, top_k=(1,)):
    # Computes the precision@k for the specified values of k
    output = F.softmax(logit, dim=1)
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def basic_accuracy(predict, answer) -> float:
    acc = (answer == predict).float().mean()
    return acc


def to_np(x):
    return x.detach().cpu().numpy()


def get_subdirs(p):
    ex_folders = []
    for it in os.scandir(p):
        if it.is_dir():
            # it.name or it.path
            ex_folders.append(it.path)

    return ex_folders


def load_tokenizer(tokenizer_type):
    if tokenizer_type == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif tokenizer_type == 'code-parser':
        tokenizer = CodeTokenizer(lang='c')
    else:
        raise ValueError(f'Unknown tokenizer type:{tokenizer_type}. Type must be (bert-base-uncased / code-parser).')

    return tokenizer


def read_jsonl_data(data_path: str):
    """
    jsonl 파일을 읽어서 return
    """
    json_data = []
    with open(data_path, 'r') as f:
        for line in f:
            json_line = json.loads(line.strip())
            json_data.append(json_line)

    return json_data


def prepare_sequence(seq, word_to_idx):
    indices = []

    for word in seq:
        if word not in word_to_idx.keys():
            indices.append(word_to_idx['<unk>'])
        else:
            indices.append(word_to_idx[word])

    # indices = np.array(indices)

    return indices


def build_tok_vocab(tokenize_target: List, tokenizer,
                    min_freq: int = 3, max_vocab=29998) -> Tuple[List[str], Dict]:
    """
    데이터 입력 받아서 vocab set return
    :param tokenize_target:
    :param tokenizer:
    :param min_freq: vocab set 최소 회수
    :param max_vocab: vocab set 최대 크기
    :return: (단어, idx)으로 이루어진 Tuple
    """
    vocab = []
    logger.debug('start parsing vocabulary set!')
    for i, target in enumerate(tokenize_target):
        try:
            temp = tokenizer.tokenize(target)
            vocab.extend(temp)

        except Exception as e_msg:
            error_target = f'idx: {i} \t target:{target}'
            logger.warning(error_target, e_msg)

    logger.debug('vocabulary set parsing done')
    logger.debug('start configuring vocabulary set!')
    vocab = collections.Counter(vocab)
    temp = {}
    # min_freq보다 적은 단어 거르기
    for key in vocab.keys():
        if vocab[key] >= min_freq:
            temp[key] = vocab[key]
    vocab = temp

    # 가장 많이 등장하는 순으로 정렬한 후, 적게 나온것 위주로 vocab set에서 빼기
    vocab = sorted(vocab, key=lambda x: -vocab[x])
    if len(vocab) > max_vocab:
        vocab = vocab[:max_vocab]

    tok2idx = {'<pad>': 0, '<unk>': 1}
    for tok in vocab:
        tok2idx[tok] = len(tok2idx)
    vocab.extend(['<pad>', '<unk>'])

    logger.debug('vocabulary set configuring done')

    return vocab, tok2idx


if __name__ == '__main__':
    p = '../logs/news/coteaching_plus/symmetric_0.2_seed4_ll_2022-11-03-19-58'
    get_subdirs(p)
