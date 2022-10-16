import os
import torch
import numpy as np
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForSequenceClassification

from utils import arg_parse, gen_forget_rate, adjust_learning_rate
from data import NewsGroups, NewsGroupsForBert
from model import NewsNet, NewsNetCNN, NewsNetLSTM, BertClassifier
from trainer import NewsGroupTrainer
from model import NewsNetVDCNN

import logging.config
import json

config = json.load(open('./logger.json'))
logging.config.dictConfig(config)

logger = logging.getLogger(__name__)


def main():
    # argument parsing
    args = arg_parse()

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
    if args.dataset == 'news':
        init_epoch = 0
        train_dataset = NewsGroups(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   noise_type=args.noise_type,
                                   noise_rate=args.noise_rate
                                   )
        test_dataset = NewsGroups(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  noise_type=args.noise_type,
                                  noise_rate=args.noise_rate
                                  )
        num_classes = train_dataset.num_classes

    elif args.dataset =='news_bert':
        init_epoch = 0
        train_dataset = NewsGroupsForBert(root='./data/',
                                          train=True,
                                          noise_type=args.noise_type,
                                          noise_rate=args.noise_rate,
                                          max_len=512
                                          )
        test_dataset = NewsGroupsForBert(root='./data/',
                                         train=False,
                                         noise_type=args.noise_type,
                                         noise_rate=args.noise_rate,
                                         max_len=512
                                         )
        num_classes = train_dataset.num_classes

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
    if args.model1 == 'fcn':
        clf1 = NewsNet(weights_matrix=train_dataset.weights_matrix, num_classes=num_classes)
    elif args.model1 == 'cnn':
        clf1 = NewsNetCNN(weights_matrix=train_dataset.weights_matrix, num_classes=num_classes)
    elif args.model1 == 'lstm':
        clf1 = NewsNetLSTM(weights_matrix=train_dataset.weights_matrix, num_classes=num_classes)
    elif args.model1 == 'vdcnn':
        clf1 = NewsNetVDCNN(weights_matrix=train_dataset.weights_matrix, num_classes=num_classes)
    elif args.model1 == 'bert':
        # clf1 = BertForSequenceClassification.from_pretrained(
        #     "bert-base-multilingual-cased",
        #     num_labels=num_classes,
        #     output_attentions=False,
        #     output_hidden_states=False
        # )
        clf1 = BertClassifier.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False
        )
    else:
        raise Exception(f'Unknown model name {args.model1}')

    clf1.cuda()
    logger.info(clf1.parameters)
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)

    # build model 2
    if args.model2 == 'fcn':
        clf2 = NewsNet(weights_matrix=train_dataset.weights_matrix, num_classes=num_classes)
    elif args.model2 == 'cnn':
        clf2 = NewsNetCNN(weights_matrix=train_dataset.weights_matrix, num_classes=num_classes)
    elif args.model2 == 'lstm':
        clf2 = NewsNetLSTM(weights_matrix=train_dataset.weights_matrix, num_classes=num_classes)
    elif args.model2 == 'vdcnn':
        clf2 = NewsNetVDCNN(weights_matrix=train_dataset.weights_matrix, num_classes=num_classes)
    elif args.model2 == 'bert':
        # clf2 = BertForSequenceClassification.from_pretrained(
        #     "bert-base-multilingual-cased",
        #     num_labels=num_classes,
        #     output_attentions=False,
        #     output_hidden_states=False
        # )
        clf2 = BertClassifier.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False
        )
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
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model_str = f'{args.noise_type}_{str(args.noise_rate)}_seed{args.seed}_{now_time}'
    writer = SummaryWriter(f'{save_dir}/{model_str}')

    # training
    ng_trainer = NewsGroupTrainer(model1=clf1,
                                  optimizer1=optimizer1,
                                  model2=clf2,
                                  optimizer2=optimizer2,
                                  device=args.device,
                                  init_epoch=init_epoch,
                                  output_dir='',
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
        train_result, _ = ng_trainer.train(n_epoch=epoch)

        # evaluate
        eval_result, _ = ng_trainer.evaluate(model_summary_path='', n_epoch=epoch, mode='test')

        # save results
        """
        writer.add_scalars(main_tag='Loss/train_eval',
                           global_step=epoch,
                           tag_scalar_dict={'train_loss': train_result['Avg loss'],
                                            'val_loss': eval_result['Avg loss']})
        writer.add_scalars(main_tag='Acc/train_eval_test',
                           global_step=epoch,
                           tag_scalar_dict={'train_loss': train_result['Avg acc'],
                                            'val_loss': eval_result['Acc'],
                                            'test_loss': test_result['Acc']})
        writer.add_scalars(main_tag='F1/train_eval_test',
                           global_step=epoch,
                           tag_scalar_dict={'train_loss': train_result['Avg f1'],
                                            'val_loss': eval_result['F1'],
                                            'test_loss': test_result['Acc']})
        """

    writer.close()


if __name__ == '__main__':
    main()

