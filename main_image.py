import os
import torch
import numpy as np
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import arg_parse, gen_forget_rate, adjust_learning_rate
from data import CIFAR10
from model import SmallCNN
from trainer import NewsGroupTrainer

import logging.config
import json

config = json.load(open('./logger.json'))
logging.config.dictConfig(config)

logger = logging.getLogger(__name__)


def count_parameters(model, trainable=False):
    if trainable is True:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def ex_image_cnn_cnn(args):
    args.dataset = 'cifar10'
    args.n_epoch = 200
    args.noise_type = 'symmetric'
    args.noise_rate = 0.2
    args.model1 = 'cnn'
    args.model2 = 'cnn'
    args.model_type = 'coteaching_plus'

    lst_seed = [1, 2, 3, 4, 5]
    lst_kernels = [[3, 3], [5, 5], [7, 7]]

    for m in range(len(lst_kernels)):
        for n in range(m, len(lst_kernels)):
            for s in lst_seed:
                args.seed = s
                args.cnn_opt1 = lst_kernels[m]
                args.cnn_opt2 = lst_kernels[n]

                main(args)


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
    num_classes = 0
    if args.dataset == 'cifar10':
        init_epoch = args.init_epoch
        num_classes = 10

        train_dataset = CIFAR10(root='./data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                                )

        test_dataset = CIFAR10(root='./data/',
                               train=False,
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                               )

    else:
        raise Exception(f'Unknown dataset {args.dataset}')

    # load dataloader
    logger.info('Loading dataset...')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              drop_last=True,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             drop_last=True,
                             shuffle=False)

    # build model 1
    logger.info('Building model...')
    if args.model1 == 'cnn':
        clf1 = SmallCNN(kernel_size=args.cnn_opt1, num_classes=num_classes)
    else:
        raise Exception(f'Unknown model name {args.model1}')

    clf1.cuda()
    logger.info(clf1.parameters)
    logger.info(f'model parameters(trainable/all): {count_parameters(clf1, trainable=True)} / {count_parameters(clf1)}')

    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)

    # build model 2
    if args.model2 == 'cnn':
        clf2 = SmallCNN(kernel_size=args.cnn_opt2, num_classes=num_classes)
    else:
        raise Exception(f'Unknown model name {args.model2}')

    clf2.cuda()
    logger.info(clf2.parameters)
    logger.info(f'model parameters(trainable/all): {count_parameters(clf2, trainable=True)} / {count_parameters(clf2)}')
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
    ng_trainer = NewsGroupTrainer(model1=clf1,
                                  optimizer1=optimizer1,
                                  model2=clf2,
                                  optimizer2=optimizer2,
                                  device=args.device,
                                  init_epoch=init_epoch,
                                  output_dir=f'{save_dir}/{model_str}',
                                  train_loader=train_loader,
                                  model_type=args.model_type,
                                  rate_schedule=rate_schedule,
                                  noise_or_not=train_dataset.noise_or_not,
                                  test_loader=test_loader)

    logger.info('Start train & evaluate')
    for epoch in range(args.n_epoch):
        logger.info(f'Epoch [{epoch}/{args.n_epoch}]')
        # adjust learning rate
        adjust_learning_rate(optimizer=optimizer1, epoch=epoch, alpha_plan=alpha_plan, beta1_plan=beta1_plan)
        adjust_learning_rate(optimizer=optimizer2, epoch=epoch, alpha_plan=alpha_plan, beta1_plan=beta1_plan)

        # train
        train_result, model_summary_path = ng_trainer.train(n_epoch=epoch)

        # evaluate
        test_result, _ = ng_trainer.evaluate(model_summary_path=model_summary_path, n_epoch=epoch, mode='test')

        # save results
        writer.add_scalars(main_tag='Loss/train_test',
                           global_step=epoch,
                           tag_scalar_dict={'train_loss': train_result['Avg loss'],
                                            'val_loss': test_result['loss']})
        writer.add_scalars(main_tag='Acc/train_test',
                           global_step=epoch,
                           tag_scalar_dict={'train_acc': train_result['Avg acc'],
                                            'test_acc': test_result['Acc']})

    writer.close()


if __name__ == '__main__':
    args = arg_parse()
    ex_image_cnn_cnn(args)