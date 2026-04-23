import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch #looksam
from torch.optim import Optimizer #looksam



class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class ApproximateSAM(Optimizer):
    """
    Single-pass (approximate) SAM (LookSAM).
    We do:
      1) forward/backward to get grad(g) at w
      2) w+ = w + rho*g/||g||
      3) approximate grad(w+) ~ grad(w)
      4) revert w, then apply that approximate grad
    """
    def __init__(self, params, base_optimizer_cls, lr=0.1, momentum=0.9, weight_decay=0.0, rho=0.05, **kwargs):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Single-step approximate SAM:
          - We assume we already did one forward/backward pass externally,
            so p.grad stores grad(w).
          - Perturb w -> w+
          - approximate grad(w+) = grad(w)
          - revert w, then do the final step with approximate grad
        """
        loss = None
        if closure is not None:
            loss = closure()

        # gather original grads
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.clone())

        # compute global or per-parameter norm
        rho = self.param_groups[0]['rho']  # assume a single rho
        grad_norm = torch.norm(torch.stack([g.norm(p=2) for g in grads]), p=2) + 1e-12

        # 1) Perturb w -> w+
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.add_(grads[idx], alpha=(rho / grad_norm))
                    idx += 1

        # 2) Approx grad(w+) ~ grad(w), so we keep the old grads, but we must revert w
        # revert w, then we will apply the same grad as final update
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.sub_(grads[idx], alpha=(rho / grad_norm))
                    idx += 1

        # 3) Now we apply the old grad to w
        # Overwrite p.grad with the old grad again
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.copy_(grads[idx])
                    idx += 1

        # 4) step with base optimizer
        self.base_optimizer.step()

        return loss
