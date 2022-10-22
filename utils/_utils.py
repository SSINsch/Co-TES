import argparse
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, help='now only support news dataset', default='news')
    parser.add_argument('--model1', type=str, help='first model type')
    parser.add_argument('--model2', type=str, help='secibd model type')
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
