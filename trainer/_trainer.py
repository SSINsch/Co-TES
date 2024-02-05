from sklearn.metrics import f1_score
import os
import torch

from utils import to_np
from utils import basic_accuracy
from loss import loss_coteaching, loss_coteaching_plus
import logging

logger = logging.getLogger(__name__)


class NewsGroupTrainer:
    def __init__(self, model1, optimizer1, model2, optimizer2,
                 device,
                 init_epoch,
                 output_dir,
                 train_loader,
                 model_type,
                 rate_schedule,
                 noise_or_not,
                 dataset='news',
                 test_loader=None,
                 eval_loader=None):
        self.model1 = model1
        self.optimizer1 = optimizer1
        self.model2 = model2
        self.optimizer2 = optimizer2
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.init_epoch = init_epoch
        self.model_type = model_type
        self.rate_schedule = rate_schedule
        self.noise_or_not = noise_or_not
        self.dataset = dataset

    def save_model(self, n_epoch, avg_loss, accuracy, macro_f1_score, mode):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        model1_to_save = self.model1.module if hasattr(self.model1, 'module') else self.model1
        model2_to_save = self.model2.module if hasattr(self.model2, 'module') else self.model2
        postfix1 = f'model1_{self.model1.alg_name}_{mode}_{n_epoch:03d}.bin'
        postfix2 = f'model2_{self.model2.alg_name}_{mode}_{n_epoch:03d}.bin'
        output_path1 = os.path.join(self.output_dir, postfix1)
        output_path2 = os.path.join(self.output_dir, postfix2)

        if mode == 'train':
            torch.save({
                'mode': mode,
                'epoch': n_epoch,
                'model_state_dict': model1_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer1.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'f1-score': macro_f1_score
            }, output_path1)
            torch.save({
                'mode': mode,
                'epoch': n_epoch,
                'model_state_dict': model2_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer2.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'f1-score': macro_f1_score
            }, output_path2)
        else:
            torch.save({
                'mode': mode,
                'epoch': n_epoch,
                'loss': avg_loss,
                'accuracy': accuracy,
                'f1-score': macro_f1_score
            }, output_path1)
            torch.save({
                'mode': mode,
                'epoch': n_epoch,
                'loss': avg_loss,
                'accuracy': accuracy,
                'f1-score': macro_f1_score
            }, output_path2)

        logger.info(f"Saving model checkpoint to as {output_path1} and {output_path2}")

        return output_path1, output_path2

    def load_model(self, model_path):
        model_path1, model_path2 = model_path
        checkpoint = torch.load(model_path1)
        self.model1.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint = torch.load(model_path2)
        self.model2.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Load model checkpoint from {model_path1} and {model_path2}")

    def train(self, n_epoch, save=False):
        self.model1.train()
        self.model2.train()

        total_step = len(self.train_loader)
        total_loss_1, total_acc_1 = 0, 0
        total_loss_2, total_acc_2 = 0, 0
        for step, (data, labels, indexes) in enumerate(self.train_loader):
            # to cpu / to gpu
            ind = indexes.cpu().numpy().transpose()
            if self.dataset == 'news':
                data = data.long().to(self.device)
                labels = labels.to(self.device)
            elif self.dataset == 'cifar10':
                data = data.float().to(self.device)
                labels = labels.to(self.device).long()
            else:
                raise ValueError(f'Unknown dataset {self.dataset}')

            # initialize optimizer
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()

            # to model 1
            logits_1 = self.model1(data)
            max_predictions, argmax_predictions = logits_1.max(1)
            acc_1 = basic_accuracy(argmax_predictions, labels)
            acc_1 = to_np(acc_1)
            total_acc_1 += acc_1

            # to model 2
            logits_2 = self.model2(data)
            max_predictions, argmax_predictions = logits_2.max(1)
            acc_2 = basic_accuracy(argmax_predictions, labels)
            acc_2 = to_np(acc_2)
            total_acc_2 += acc_2

            # Todo. 이 아래 Loss 들의 distribution 찍어보기
            # get loss
            if n_epoch < self.init_epoch:
                loss_1, loss_2, _, _ = loss_coteaching(logits_1, logits_2, labels,
                                                       self.rate_schedule[n_epoch], ind,
                                                       self.noise_or_not)
            else:
                if self.model_type == 'coteaching_plus':
                    loss_1, loss_2, _, _ = loss_coteaching_plus(logits_1, logits_2, labels,
                                                                self.rate_schedule[n_epoch], ind,
                                                                self.noise_or_not, n_epoch * step)
                elif self.model_type == 'coteaching':
                    loss_1, loss_2, _, _ = loss_coteaching(logits_1, logits_2, labels,
                                                           self.rate_schedule[n_epoch], ind,
                                                           self.noise_or_not)
                else:
                    raise Exception('Init Epoch is something wrong...')

            total_loss_1 += loss_1.item()
            total_loss_2 += loss_2.item()

            # backpropagation
            loss_1.backward()
            self.optimizer1.step()

            loss_2.backward()
            self.optimizer2.step()

            # print result
            if (step % 10 == 0) or ((step+1) == total_step):
                log_message = f"\tStep [{step + 1:03}/{total_step}] "
                log_message += f'Loss_1: {loss_1.item():.4f}, Loss_2: {loss_2.item():.4f}, '
                log_message += f'Acc_1: {acc_1:.4f}, Acc_2: {acc_2:.4f}, '
                # log_message += f'F1-score_1: {0}, F1-score_2: {0}'
                logger.info(log_message)

        avg_acc_1 = total_acc_1 / total_step
        avg_acc_2 = total_acc_2 / total_step
        avg_loss_1 = total_loss_1 / total_step
        avg_loss_2 = total_loss_2 / total_step

        if avg_acc_1 > avg_acc_2:
            avg_acc = avg_acc_1
            avg_loss = avg_loss_1
            avg_f1 = 0
        else:
            avg_acc = avg_acc_2
            avg_loss = avg_loss_2
            avg_f1 = 0

        if save is True:
            model_summary_path = self.save_model(n_epoch, avg_loss, avg_acc, avg_f1, mode='train')
        else:
            model_summary_path = ''

        train_result = {'Avg acc': avg_acc, 'Avg f1-score': avg_f1, 'Avg loss': avg_loss}

        return train_result, model_summary_path

    def evaluate(self, model_summary_path, n_epoch, mode='test', load=False):
        # mode check
        if (mode == 'test') and (self.test_loader is not None):
            loader = self.test_loader
        elif (mode == 'eval') and (self.eval_loader is not None):
            loader = self.eval_loader
        else:
            raise ValueError('evaluate mode not found')
        total_step = len(loader)

        if load is True:
            self.load_model(model_summary_path)
        self.model1.eval()
        self.model2.eval()

        argmax_labels_list = []
        argmax_predictions_list_1, argmax_predictions_list_2 = [], []
        total_loss_1, total_loss_2 = 0, 0

        with torch.no_grad():
            for step, (data, labels, indexes) in enumerate(loader):
                # to cpu/gpu
                ind = indexes.cpu().numpy().transpose()
                if self.dataset == 'news':
                    data = data.long().to(self.device)
                    labels = labels.to(self.device)
                elif self.dataset == 'cifar10':
                    data = data.float().to(self.device)
                    labels = labels.to(self.device).long()
                else:
                    raise ValueError(f'Unknown dataset {self.dataset}')

                # to model
                logits_1 = self.model1(data)
                max_predictions_1, argmax_predictions_1 = logits_1.max(1)
                logits_2 = self.model2(data)
                max_predictions_2, argmax_predictions_2 = logits_2.max(1)

                # get loss
                if n_epoch < self.init_epoch:
                    loss_1, loss_2, _, _ = loss_coteaching(logits_1, logits_2, labels,
                                                           self.rate_schedule[n_epoch], ind,
                                                           self.noise_or_not)
                else:
                    if self.model_type == 'coteaching_plus':
                        loss_1, loss_2, _, _ = loss_coteaching_plus(logits_1, logits_2, labels,
                                                                    self.rate_schedule[n_epoch], ind,
                                                                    self.noise_or_not, n_epoch * step)
                    elif self.model_type == 'coteaching':
                        loss_1, loss_2, _, _ = loss_coteaching(logits_1, logits_2, labels,
                                                               self.rate_schedule[n_epoch], ind,
                                                               self.noise_or_not)
                    else:
                        loss_1, loss_2 = None, None
                        raise Exception('Init Epoch is something wrong...')

                total_loss_1 += loss_1.item()
                total_loss_2 += loss_2.item()

                argmax_labels_list.append(labels)
                argmax_predictions_list_1.append(argmax_predictions_1)
                argmax_predictions_list_2.append(argmax_predictions_2)

        # Acc
        argmax_labels = torch.cat(argmax_labels_list, 0)
        argmax_predictions_1 = torch.cat(argmax_predictions_list_1, 0)
        argmax_predictions_2 = torch.cat(argmax_predictions_list_2, 0)
        acc_1 = basic_accuracy(argmax_predictions_1, argmax_labels)
        acc_1 = to_np(acc_1)
        acc_2 = basic_accuracy(argmax_predictions_2, argmax_labels)
        acc_2 = to_np(acc_2)

        # f1 score
        argmax_labels_np_array = to_np(argmax_labels)
        argmax_predictions_np_array_1 = to_np(argmax_predictions_1)
        argmax_predictions_np_array_2 = to_np(argmax_predictions_2)
        macro_f1_score_1 = f1_score(argmax_labels_np_array, argmax_predictions_np_array_1, average='macro')
        macro_f1_score_2 = f1_score(argmax_labels_np_array, argmax_predictions_np_array_2, average='macro')

        avg_loss_1 = total_loss_1 / total_step
        avg_loss_2 = total_loss_2 / total_step

        # print result
        log_message = f'\t[Test results] Loss_1: {loss_1.item():.4f}, Loss_2: {loss_2.item():.4f}, '
        log_message += f'Acc_1: {acc_1:.4f}, Acc_2: {acc_2:.4f}, '
        log_message += f'F1-score_1: {macro_f1_score_1:.4f}, F1-score_2: {macro_f1_score_2:.4f}'
        logger.info(log_message)

        if acc_1 > acc_2:
            avg_acc = acc_1
            avg_loss = avg_loss_1
            avg_f1 = 0
        else:
            avg_acc = acc_2
            avg_loss = avg_loss_2
            avg_f1 = 0

        result = {'Acc': avg_acc, 'f1-score': avg_f1, 'loss': avg_loss}

        return result, model_summary_path
