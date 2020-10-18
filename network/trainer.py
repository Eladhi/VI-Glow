import os
import time
import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from misc import util, ops
from network.model import Glow


class Trainer:
    criterion_dict = {
        'single_class': lambda y_logits, y: Glow.single_class_loss(y_logits, y),
        'multi_class': lambda y_logits, y_onehot: Glow.single_class_loss(y_logits, y_onehot)
    }

    def __init__(self, hps, result_subdir,
                 step, graph, optimizer, scheduler, devices,
                 dataset, data_device):
        """
        Network trainer

        :param hps: hyper-parameters for this network
        :type hps: dict
        :param result_subdir: path to result sub-directory
        :type result_subdir: str
        :param step: global step of model
        :type step: int
        :param graph: model graph
        :type graph: torch.nn.Module
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param scheduler: learning rate scheduler
        :type scheduler: function
        :param devices: list of usable devices for model running
        :type devices: list
        :param dataset: dataset for training model
        :type dataset: torch.utils.data.Dataset
        :param data_device:
        :type data_device:
        """
        # general
        self.hps = hps
        self.result_subdir = result_subdir
        self.start_time = time.time()
        # state
        self.step = step
        self.graph = graph
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.devices = devices
        # data
        self.data_device = data_device
        self.batch_size = self.hps.optim.num_batch_train
        self.num_classes = self.hps.dataset.num_classes
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size,
                                      num_workers=self.hps.dataset.num_workers,
                                      shuffle=True,
                                      drop_last=True)
        self.num_epochs = self.hps.optim.num_epochs
        # ablation
        self.y_condition = self.hps.ablation.y_condition
        if self.y_condition:
            self.y_criterion = self.hps.ablation.y_criterion
            assert self.y_criterion in self.criterion_dict.keys(), "Unsupported criterion: {}".format(self.y_criterion)
        self.max_grad_clip = self.hps.ablation.max_grad_clip
        self.max_grad_norm = self.hps.ablation.max_grad_norm
        # logging
        self.writer = SummaryWriter(log_dir=self.result_subdir)
        self.interval_scalar = self.hps.optim.interval_scalar
        self.interval_snapshot = self.hps.optim.interval_snapshot
        self.interval_valid = self.hps.optim.interval_valid
        self.interval_sample = self.hps.optim.interval_sample
        self.num_sample = self.hps.optim.num_sample

    def train(self):
        """
        Train network
        """
        self.graph.train()

        for epoch in range(self.num_epochs):
            print('[Trainer] Epoch ({}/{})'.format(epoch, self.num_epochs))
            progress = tqdm(self.data_loader)
            for idx, batch in enumerate(progress):
                # update learning rate
                lr = self.scheduler(global_step=self.step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.optimizer.zero_grad()
                if self.step % self.interval_scalar == 0 and self.step > 0:
                    self.writer.add_scalar('lr/lr', lr, self.step)

                # extract batch data - may ignore class annotations
                batch[0] = batch[0].to(self.data_device)
                x = batch[0]

                y = None
                y_onehot = None
                if self.y_condition:
                    if self.y_criterion == 'single_class':
                        assert 'y' in batch.keys(), 'Single-class criterion needs "y" in batch data'
                        y = batch['y']
                        y_onehot = ops.onehot(y, self.num_classes)
                    else:
                        assert 'y_onehot' in batch.keys(), 'Multi-class criterion needs "y_onehot" in batch data'
                        y_onehot = batch['y_onehot']

                # initialize actnorm layer at first
                if self.step == 0:
                    self.graph(x=x[:self.batch_size // len(self.devices), ...],
                               y_onehot=y_onehot[:self.batch_size // len(self.devices), ...]
                               if y_onehot is not None else None)
                # data parallel
                if len(self.devices) > 1 and not hasattr(self.graph, 'module'):
                    self.graph = torch.nn.parallel.DataParallel(module=self.graph,
                                                                device_ids=self.devices,
                                                                output_device=self.devices[0])

                # forward model
                z, nll, y_logits = self.graph(x=x, y_onehot=y_onehot)

                # compute loss
                generative_loss = Glow.generative_loss(nll)
                classification_loss = 0
                if self.y_condition:
                    classification_loss = self.criterion_dict[self.y_criterion](y_logits,
                                                                                y if self.y_criterion == 'single_class' else y_onehot)
                loss = generative_loss + classification_loss * self.hps.model.weight_y
                if self.step % self.interval_scalar == 0 and self.step > 0:
                    self.writer.add_scalar('loss/generative_loss', generative_loss, self.step)
                    if self.y_condition:
                        self.writer.add_scalar('loss/classification_loss', classification_loss, self.step)

                # backward model
                self.graph.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                # gradient operation
                if self.max_grad_clip is not None and self.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                    if self.step % self.interval_scalar == 0 and self.step > 0:
                        self.writer.add_scalar("grad_norm/grad_norm", grad_norm, self.step)

                # optimize
                self.optimizer.step()

                # snapshot
                if self.step % self.interval_snapshot == 0 and self.step > 0:
                    util.save_model(result_subdir=self.result_subdir,
                                    step=self.step,
                                    graph=self.graph,
                                    optimizer=self.optimizer,
                                    seconds=time.time() - self.start_time,
                                    is_best=True)

                with torch.no_grad():
                    # valid
                    if self.step % self.interval_valid == 0 and self.step > 0:
                        img = self.graph(z=z, y_onehot=y_onehot, reverse=True)
                        for i in range(min(self.num_sample, img.shape[0])):
                            self.writer.add_image("reconstructed/{}".format(i),
                                                  ops.cat_channel(img[i]/2+0.5, batch[0][i]/2+0.5),
                                                  self.step)

                    # sample
                    if self.step % self.interval_sample == 0 and self.step > 0:
                        img = self.graph(z=None, y_onehot=y_onehot, eps_std=0.7, reverse=True)
                        for i in range(min(self.num_sample, img.shape[0])):
                            self.writer.add_image("sample/{}".format(i),
                                                  img[i]/2+0.5, self.step)

                self.step += 1

        self.writer.export_scalars_to_json(os.path.join(self.result_subdir, "all_scalars.json"))
        self.writer.close()
