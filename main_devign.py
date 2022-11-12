import os
import torch
import numpy as np
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import arg_parse, gen_forget_rate, adjust_learning_rate
from utils import load_tokenizer, read_jsonl_data, build_tok_vocab
from data.code_vuln import get_loader
from data import NewsGroups
from model import NewsNet, NewsNetCNN, NewsNetLSTM, DevignBiLSTM
from trainer import NewsGroupTrainer, DevignTrainer
from model import NewsNetVDCNN

import logging.config
import json

config = json.load(open('./logger.json'))
logging.config.dictConfig(config)

logger = logging.getLogger(__name__)


def main(args):
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # parameters
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.lr
    if args.forget_rate is None:
        forget_rate = args.noise_rate
    else:
        forget_rate = args.forget_rate

    # Adjust learning rate and betas for Adam Optimizer
    momentum_1 = 0.9
    momentum_2 = 0.1
    alpha_plan = [learning_rate] * args.n_epoch
    beta1_plan = [momentum_1] * args.n_epoch
    # decay epoch 부터 last epoch까지 점점 learning rate를 조절
    for i in range(args.epoch_decay_start, args.n_epoch):
        alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
        beta1_plan[i] = momentum_2
    rate_schedule = gen_forget_rate(args.n_epoch, args.num_gradual, forget_rate, args.fr_type)

    # load dataset
    if args.dataset == 'devign':
        tokenizer = load_tokenizer(tokenizer_type='code-parser')
        init_epoch = 0

        # vocab set
        # data = (project, commit_id, target, func)
        path_train = './data/devign/train.jsonl'
        path_valid = './data/devign/valid.jsonl'
        path_test = './data/devign/test.jsonl'

        data = read_jsonl_data(path_train)
        functions = [x['func'] for x in data]
        func_tok_vocab_set, func_tok2idx = build_tok_vocab(functions, tokenizer, min_freq=1)
        logger.info(f'Vocab set size: {len(func_tok2idx)}')

        logger.info('Loading dataset...')
        train_loader, num_train = get_loader(data_path=path_train,
                                             batch_size=batch_size,
                                             tokenizer=tokenizer,
                                             tok2idx=func_tok2idx,
                                             block_size=1000)

        eval_loader, num_eval = get_loader(data_path=path_valid,
                                           batch_size=batch_size,
                                           tokenizer=tokenizer,
                                           tok2idx=func_tok2idx,
                                           block_size=1000)

        test_loader, num_test = get_loader(data_path=path_test,
                                           batch_size=batch_size,
                                           tokenizer=tokenizer,
                                           tok2idx=func_tok2idx,
                                           block_size=1000)

        num_classes = 2

    else:
        raise Exception(f'Unknown dataset {args.dataset}')

    # build model 1
    logger.info('Building model...')
    if args.model1 == 'lstm':
        hidden_dim = 256
        embed_size = 256
        num_lstm_layer = 2
        clf1 = DevignBiLSTM(hidden_dim=hidden_dim,
                            num_lstm_layer=num_lstm_layer,
                            n_classes=num_classes,
                            embed_size=embed_size,
                            vocab_size=len(func_tok2idx))
    else:
        raise Exception(f'Unknown model name {args.model1}')

    clf1.cuda()
    logger.info(clf1.parameters)
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)

    # build model 2
    if args.model2 == 'lstm':
        hidden_dim = 256
        embed_size = 256
        num_lstm_layer = 2
        clf2 = DevignBiLSTM(hidden_dim=hidden_dim,
                            num_lstm_layer=num_lstm_layer,
                            n_classes=num_classes,
                            embed_size=embed_size,
                            vocab_size=len(func_tok2idx))
    else:
        raise Exception(f'Unknown model name {args.model2}')

    clf2.cuda()
    logger.info(clf2.parameters)
    optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate)

    # set result folder and file
    save_dir = args.result_dir + '\\' + args.dataset + '\\' + args.model_type
    logger.info(f'log directory : {save_dir}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    model_str = f'{args.noise_type}_{str(args.noise_rate)}_seed{args.seed}_{args.model1[0]}{args.model2[0]}_{now_time}'
    writer = SummaryWriter(f'{save_dir}/{model_str}')
    with open(f'{save_dir}/{model_str}/info.json', "w") as f:
        json.dump(args.__dict__, f)

    # training
    noise_or_not = np.asarray([True] * num_train)
    devign_trainer = DevignTrainer(model1=clf1,
                                   optimizer1=optimizer1,
                                   model2=clf2,
                                   optimizer2=optimizer2,
                                   device=args.device,
                                   init_epoch=init_epoch,
                                   output_dir=f'{save_dir}/{model_str}',
                                   train_loader=train_loader,
                                   model_type=args.model_type,
                                   rate_schedule=rate_schedule,
                                   noise_or_not=noise_or_not,
                                   test_loader=test_loader)

    logger.info('Start train & evaluate')
    for epoch in range(args.n_epoch):
        logger.info(f'Epoch [{epoch}/{args.n_epoch}]')
        # adjust learning rate
        adjust_learning_rate(optimizer=optimizer1, epoch=epoch, alpha_plan=alpha_plan, beta1_plan=beta1_plan)
        adjust_learning_rate(optimizer=optimizer2, epoch=epoch, alpha_plan=alpha_plan, beta1_plan=beta1_plan)

        # train
        train_result, model_summary_path = devign_trainer.train(n_epoch=epoch)

        # evaluate
        test_result, _ = devign_trainer.evaluate(model_summary_path=model_summary_path, n_epoch=epoch, mode='test')

        # save results
        writer.add_scalars(main_tag='Loss/train_test',
                           global_step=epoch,
                           tag_scalar_dict={'train_loss': train_result['Avg loss'],
                                            'test_loss': test_result['loss']})
        writer.add_scalars(main_tag='Acc/train_test',
                           global_step=epoch,
                           tag_scalar_dict={'train_acc': train_result['Avg acc'],
                                            'test_acc': test_result['Acc']})

    writer.close()


if __name__ == '__main__':

    args = arg_parse()
    lst_seed = [1, 2, 3, 4, 5]
    epochs = 100

    for s in lst_seed:
        args.seed = s
        args.n_epoch = epochs
        args.batch_size = 64
        args.dataset = 'devign'
        args.model1 = 'lstm'
        args.model2 = 'lstm'
        args.noise_rate = 0

        main(args)

