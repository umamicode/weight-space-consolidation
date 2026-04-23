#Replay-aware Adaptive Minimization.
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.optim import Optimizer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 70
lrate = 0.1
milestones = [30, 50]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 4
T = 2

###############################################################################
# 1) Define the Replay-Aware Adaptive Moment (RAM) Optimizer
###############################################################################
class RAM(Optimizer):
    """
    Replay-Aware Adaptive Moment (RAM) Optimizer.
    We assume we can pass two sets of gradients:
      - grad_new (gradient from current / new data)
      - grad_mem (gradient from replay memory)

    conflict_threshold controls whether to project new gradient if they conflict.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0.0, conflict_threshold=-0.5):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        conflict_threshold=conflict_threshold)
        super(RAM, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, grad_new, grad_mem):
        """
        grad_new: dict of p -> gradient from new data
        grad_mem: dict of p -> gradient from replay memory
        """
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            threshold = group['conflict_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue
                # Get the separate gradients
                g_n = grad_new[p]   # shape = same as p
                g_m = grad_mem[p]
                if g_n is None or g_m is None:
                    # Fallback: if we couldn't split, skip or treat as standard grad
                    continue

                # Cosine similarity
                denom_nm = (g_n.norm() * g_m.norm() + eps)
                if denom_nm > 0:
                    cos_sim = torch.dot(g_n.flatten(), g_m.flatten()) / denom_nm
                else:
                    cos_sim = 1.0  # no conflict if either is zero

                # If conflicting, project new gradient
                if cos_sim < threshold:
                    dot_nm = torch.dot(g_n.flatten(), g_m.flatten())
                    denom_m = g_m.norm().pow(2) + eps
                    proj_coeff = dot_nm / denom_m
                    g_n = g_n - proj_coeff * g_m

                # Final combined gradient
                g = g_n + g_m

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Weight decay
                if weight_decay != 0:
                    g = g.add(p, alpha=weight_decay)

                # Adam's moment updates
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr

                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)

###############################################################################
# 2) Our main Replay class with integrated RAM
###############################################################################
class Replay(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                      shuffle=False, num_workers=num_workers)

        # Procedure
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            # Keep original optimizer for initial training if you prefer
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
        else:
            # Example usage: we swap out SGD for RAM here
            # => or keep it as is, but let's demonstrate RAM
            ram_optimizer = RAM(
                self._network.parameters(),
                lr=lrate,        # same base LR you had for SGD
                betas=(0.9, 0.99),
                eps=1e-8,
                weight_decay=weight_decay,
                conflict_threshold=-0.5  # you can tune
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=ram_optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, ram_optimizer, scheduler)

    ############################################################################
    # UTILITY to separate new vs. memory data in each batch
    ############################################################################
    def _split_new_and_mem(self, inputs, targets):
        """
        Attempt to split a single batch into new data portion vs. memory portion.
        This depends on how you label appended memory. For instance:
          - new labels are in [self._known_classes, self._total_classes)
          - old labels are in [0, self._known_classes)
        We'll create a mask and return two subsets.
        Adjust as needed to match your dataset structure.
        """
        new_mask = (targets >= self._known_classes) & (targets < self._total_classes)
        mem_mask = (targets < self._known_classes)
        new_indices = new_mask.nonzero(as_tuple=True)[0]
        mem_indices = mem_mask.nonzero(as_tuple=True)[0]

        inputs_new = inputs[new_indices]
        targets_new = targets[new_indices]

        inputs_mem = inputs[mem_indices]
        targets_mem = targets[mem_indices]

        return (inputs_new, targets_new), (inputs_mem, targets_mem)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                optimizer.zero_grad()
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # Acc
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = (f"Task {self._cur_task}, Epoch {epoch+1}/{init_epoch} => "
                        f"Loss {losses/len(train_loader):.3f}, "
                        f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}")
            else:
                info = (f"Task {self._cur_task}, Epoch {epoch+1}/{init_epoch} => "
                        f"Loss {losses/len(train_loader):.3f}, "
                        f"Train_accy {train_acc:.2f}")

            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """
        We assume 'optimizer' might be our custom RAM. Then we do:
          1) zero_grad
          2) compute separate grads on new + mem sub-batches
          3) call optimizer.step(grad_new, grad_mem)
        """
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # Split batch into new vs. memory subsets
                (inp_new, tar_new), (inp_mem, tar_mem) = self._split_new_and_mem(inputs, targets)

                # We'll accumulate separate gradients for new data, mem data
                grad_new_dict = {}
                grad_mem_dict = {}

                # 1) ZERO all grads first (like normal)
                optimizer.zero_grad()

                ########## Compute grad_new ##########
                if len(inp_new) > 0:
                    logits_new = self._network(inp_new)["logits"]
                    loss_new = F.cross_entropy(logits_new, tar_new)
                    loss_new.backward(retain_graph=True)

                    # store the parameter.grad as grad_new
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                grad_new_dict[p] = p.grad.detach().clone()
                    # Clear grads again
                    self._network.zero_grad()

                ########## Compute grad_mem ##########
                if len(inp_mem) > 0:
                    logits_mem = self._network(inp_mem)["logits"]
                    loss_mem = F.cross_entropy(logits_mem, tar_mem)
                    loss_mem.backward(retain_graph=False)

                    # store the parameter.grad as grad_mem
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                grad_mem_dict[p] = p.grad.detach().clone()
                    # Clear grads again
                    self._network.zero_grad()

                else:
                    # If no mem data in this batch, set mem grads to zero
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            grad_mem_dict[p] = torch.zeros_like(p)

                ########## Combine total loss for logging ##########
                # The "loss" is just sum of cross entropy on new + mem
                if len(inp_new) > 0 and len(inp_mem) > 0:
                    loss_batch = loss_new.item() + loss_mem.item()
                elif len(inp_new) > 0:
                    loss_batch = loss_new.item()
                elif len(inp_mem) > 0:
                    loss_batch = loss_mem.item()
                else:
                    loss_batch = 0.0

                losses += loss_batch

                ########## RAM step ##########
                # Provide grad_new_dict & grad_mem_dict to optimizer
                optimizer.step(grad_new_dict, grad_mem_dict)

                ########## Evaluate predictions on the entire batch ##########
                # We can do a forward pass (optional) for accuracy over entire 'inputs'
                with torch.no_grad():
                    logits_all = self._network(inputs)["logits"]
                    _, preds_all = torch.max(logits_all, dim=1)
                    correct += preds_all.eq(targets).cpu().sum()
                    total += len(targets)

            # End epoch
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = (f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => "
                        f"Loss {losses/len(train_loader):.3f}, "
                        f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}")
            else:
                info = (f"Task {self._cur_task}, Epoch {epoch+1}/{epochs} => "
                        f"Loss {losses/len(train_loader):.3f}, "
                        f"Train_accy {train_acc:.2f}")

            prog_bar.set_description(info)

        logging.info(info)
