#Single-Pass Approximate SAM + SWA
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.weight_average import weight_average, stochastic_weight_average, stochastic_layer_average, ema_update
from utils.sam import SAM, enable_running_stats, disable_running_stats, ApproximateSAM

from torch.optim.swa_utils import AveragedModel, SWALR
import copy

EPSILON = 1e-8


init_epoch = 200 #200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 70 #70   #40/70/100/130/160
lrate = 0.1
milestones = [30, 50]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 4
T = 2


class Replay(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self._previous_network = None
    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # Loader
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        #We do not interfere with the initial task
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        #We will interfere when task_idx > 0
        else:
            #optimizer = optim.SGD(self._network.parameters(),lr=lrate,momentum=0.9,weight_decay=weight_decay)
            #scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)   
            
            base_optimizer = optim.SGD
            #optimizer = SAM(self._network.parameters(), base_optimizer, lr=lrate, momentum=0.9, weight_decay=weight_decay)  
            optimizer = ApproximateSAM(self._network.parameters(), base_optimizer, lr=lrate, momentum=0.9, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)   
            self._update_representation(train_loader, test_loader, optimizer,scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler=None, lambda_value=0.5):
        """
        Single-Pass Approximate SAM version
        """

        self._previous_network = copy.deepcopy(self._network.state_dict())

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, (indices, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # single forward/backward
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()

                # single-step approximate SAM
                optimizer.step()  # calls ApproximateSAM.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).sum().item()
                total += len(targets)

            if scheduler is not None:
                    scheduler.step()
            train_acc = round(correct * 100.0 / (total + 1e-8), 2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = ("Task {}, Epoch {}/{} => Loss {:.3f}, TrainAcc {:.2f}, TestAcc {:.2f}".format(
                    self._cur_task, epoch+1, epochs,
                    losses/len(train_loader), train_acc, test_acc))
            else:
                info = ("Task {}, Epoch {}/{} => Loss {:.3f}, TrainAcc {:.2f}".format(
                    self._cur_task, epoch+1, epochs,
                    losses/len(train_loader), train_acc))
            prog_bar.set_description(info)

        # BN update for SWA
        logging.info(info)


    def _create_batch_norm_loader(self, original_loader):
        """
        The original_loader can be used like this
        for i, (_, inputs, targets) in enumerate(original_loader):
            SOMETHING
        inputs.shape : torch.Size([128, 3, 32, 32])
        targets.shape : torch.Size([128])

        make a new batch_norm_loader that can be used like this:
        for i, (inputs, targets) in enumerate(batch_norm_loader):
        """
        # Initialize lists to collect all inputs and targets
        all_inputs = []
        all_targets = []

        # Iterate through original_loader to extract inputs and targets
        for _, (_, inputs, targets) in enumerate(original_loader):
            all_inputs.append(inputs)  # Append inputs #cpu
            all_targets.append(targets)  # Append targets #cpu

        # Combine all collected inputs and targets into single tensors
        all_inputs = torch.cat(all_inputs, dim=0)  # Combine along batch dimension
        all_targets = torch.cat(all_targets, dim=0)  # Combine along batch dimension

        # Wrap the combined tensors into a new dataset
        batch_norm_dataset = torch.utils.data.TensorDataset(all_inputs, all_targets)


        # Create a new DataLoader for the dataset
        batch_norm_loader = DataLoader(
            batch_norm_dataset, batch_size=128, shuffle=False, num_workers=num_workers
        )
        return batch_norm_loader
    


