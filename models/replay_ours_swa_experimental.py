# Experimental fork of the WSC trainer; the paper implementation is `models/wsc.py` (registered as model_name `wsc`).
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
from torch.optim.swa_utils import AveragedModel, SWALR
import copy

EPSILON = 1e-8

init_epoch      = 200
init_lr         = 0.1
init_milestones = [60, 120, 170]
init_lr_decay   = 0.1
init_weight_decay = 0.0005

epochs         = 70
lrate          = 0.1
milestones     = [30, 50]
lrate_decay    = 0.1
batch_size     = 128
weight_decay   = 2e-4
num_workers    = 4
T = 2

class Replay(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self._previous_network = None

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info(f"Exemplar size: {self.exemplar_size}")

    def incremental_train(self, data_manager):
        self._mom_state = {}
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info(f"Learning on {self._known_classes}-{self._total_classes}")

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train", appendent=self._get_memory()
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

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(), lr=init_lr, momentum=0.9, weight_decay=init_weight_decay
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(), lr=lrate, momentum=0.9, weight_decay=weight_decay
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

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


    # ------------------------------------------------------------------
    #  main training loop for tasks > 0  (SWA starts on val-loss plateau)
    # ------------------------------------------------------------------
    def _update_representation(self, train_loader, test_loader,
                            optimizer, scheduler=None):

        # save θ(t-1)  for later trim-and-blend
        self._previous_network = copy.deepcopy(self._network.state_dict())

        # ── plateau detection state ────────────────────────────────────
        patience      = 4          # epochs of no significant val-loss drop
        plateau_tol   = 1e-4
        best_val_loss = float("inf")
        stagnant      = 0
        swa_active    = False
        swa_model     = None                                       # to be created lazily
        swa_scheduler = None
        # ───────────────────────────────────────────────────────────────

        prog = tqdm(range(epochs))
        for epoch in prog:
            # ===== 1) training ========================================
            self._network.train()
            losses, correct, total = 0.0, 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss   = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()

                # update EMA(|g|) and EMA(g²)
                grads  = [p.grad for p in self._network.parameters() if p.requires_grad]
                params = [p for p in self._network.parameters() if p.requires_grad]
                self._moment_update(params, grads)

                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, 1)
                correct += preds.eq(targets).sum().item(); total += len(targets)

            train_loss = losses / len(train_loader)

            # ===== 2) validation ======================================
            self._network.eval()
            with torch.no_grad():
                v_losses = []
                for _, (_, v_in, v_tg) in enumerate(test_loader):
                    v_in, v_tg = v_in.to(self._device), v_tg.to(self._device)
                    v_logits   = self._network(v_in)["logits"]
                    v_losses.append(F.cross_entropy(v_logits, v_tg, reduction="mean"))
            val_loss = torch.stack(v_losses).mean().item()

            # ===== 3) plateau check ===================================
            if val_loss + plateau_tol < best_val_loss:
                best_val_loss = val_loss
                stagnant      = 0
            else:
                stagnant += 1

            start_swa = (not swa_active) and (stagnant >= patience)

            # ===== 4) SWA bookkeeping =================================
            if start_swa:
                swa_active  = True
                self._pre_swa_trim()                              # blend dormant weights
                swa_model     = AveragedModel(self._network)
                swa_scheduler = SWALR(optimizer, anneal_strategy="cos",
                                    anneal_epochs=5, swa_lr=0.1)
                logging.info(f"SWA started (epoch {epoch+1}, val-loss plateau: {val_loss:.4f})")

            if swa_active:
                swa_model.update_parameters(self._network)
                swa_scheduler.step()
            elif scheduler is not None:
                scheduler.step()

            prog.set_description(
                f"T{self._cur_task} E{epoch+1}/{epochs} "
                f"train={train_loss:.3f}  val={val_loss:.3f}"
            )

        # ===== 5) finalise SWA (BN statistics) ========================
        if swa_active:
            bn_loader = self._create_batch_norm_loader(train_loader)
            torch.optim.swa_utils.update_bn(bn_loader, swa_model, device=self._device)
            self._network.load_state_dict(swa_model.module.state_dict())



    def _create_batch_norm_loader(self, original_loader):
        all_inputs, all_targets = [], []
        for _, (_, inputs, targets) in enumerate(original_loader):
            all_inputs.append(inputs); all_targets.append(targets)
        all_inputs = torch.cat(all_inputs, dim=0); all_targets = torch.cat(all_targets, dim=0)
        ds = torch.utils.data.TensorDataset(all_inputs, all_targets)
        return DataLoader(ds, batch_size=128, shuffle=False, num_workers=num_workers)

    def _compute_hessian_diag(self, inputs, targets, samples=1):
        """
        Hutchinson diag(H) for F‑cross‑entropy via rademacher probes.
        """
        params = [p for p in self._network.parameters() if p.requires_grad]
        logits = self._network(inputs)["logits"]
        loss = F.cross_entropy(logits, targets)
        diag_est = []
        for _ in range(samples):
            v = torch.randint(0,2,(sum(p.numel() for p in params),), device=self._device, dtype=torch.float32)*2-1
            grads = torch.autograd.grad(
                loss, params, create_graph=True, allow_unused=True
            )
            # replace None grads with zeros
            grads = [g if g is not None else torch.zeros_like(p) for g,p in zip(grads, params)]
            flat_grads = torch.cat([g.flatten() for g in grads])
            hv = torch.autograd.grad(
                (flat_grads * v).sum(), params, retain_graph=True, allow_unused=True
            )
            hv = [h if h is not None else torch.zeros_like(p) for h,p in zip(hv, params)]
            flat_hv = torch.cat([h.flatten() for h in hv])
            diag_est.append(v * flat_hv)
        return torch.stack(diag_est).mean(0)

    # ------------------------------------------------------------------
    # trimming routine (unchanged interface, but calls the new scorer)
    # ------------------------------------------------------------------
    def _pre_swa_trim(self, retain_percent=20,
                    *, beta1=0.9, beta2=0.999):
        """
        Trim low-importance weights by blending them with the previous
        task’s snapshot.  Importance is |m̂|×v̂ where m̂, v̂ are the
        bias-corrected first/second moment EMAs accumulated during training.
        """
        if self._previous_network is None:
            logging.warning("No previous network; skipping trim.")
            return

        # ---- flatten all importance scores ---------------------------
        flat_scores = []
        for p in self._network.parameters():
            key = (id(p), p.shape)
            st  = self._mom_state.get(key)
            if st is None or st["t"] == 0:          # never updated → score 0
                flat_scores.append(torch.zeros(p.numel(), device=p.device))
                continue

            # bias-correction exactly like Adam
            m_hat = st["m"] / (1 - beta1 ** st["t"])
            v_hat = st["v"] / (1 - beta2 ** st["t"])

            score = m_hat.abs() * v_hat            # |g| × g² proxy
            flat_scores.append(score.flatten())

        scores_flat = torch.cat(flat_scores)        # 1-D tensor
        k_frac      = retain_percent / 100.0
        offset      = 0

        # ---- parameter-wise masking / blending ------------------------
        for name, param in self._network.named_parameters():
            old = self._previous_network.get(name)
            if old is None or param.data.shape != old.shape:
                offset += param.numel()
                continue

            n      = param.numel()
            scores = scores_flat[offset: offset + n]
            offset += n

            k      = max(1, int(n * k_frac))
            thresh = scores.topk(k, largest=True).values.min()
            mask   = (scores >= thresh).view_as(param.data)

            with torch.no_grad():
                old_data = old.to(param.data.device, dtype=param.data.dtype)
                param.data[~mask].mul_(0.5).add_(old_data[~mask], alpha=0.5)

        logging.info(f"Trimmed via accumulated grad moments (retain {retain_percent}%).")


    # ------------------------------------------------------------------
    # helpers: running moment-based importance score
    # ------------------------------------------------------------------
    def _compute_moment_score(
            self, x, y,
            *, beta1=0.9, beta2=0.999, eps=1e-8, alpha=1.0):

        params = [p for p in self._network.parameters() if p.requires_grad]

        # lazy init (now a dict keyed by param-id)
        if not hasattr(self, "_mom_state"):
            self._mom_state = {}

        # ---------- current mini-batch grads ----------
        loss = F.cross_entropy(self._network(x)["logits"], y)
        grads = torch.autograd.grad(
            loss, params, retain_graph=False, create_graph=False,
            allow_unused=True)

        # —— accumulate & score ——
        flat_scores = []
        for p, g in zip(params, grads):
            if g is None:                       # param unused in loss
                g = torch.zeros_like(p)

            key = (id(p), p.shape)              # unique across shape changes
            st  = self._mom_state.get(key)
            if st is None or st["m"].shape != p.shape:
                # (re-)initialise if first time *or* shape changed
                st = {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
                self._mom_state[key] = st

            # Adam EMA updates
            st["m"].mul_(beta1).add_(g.abs(), alpha=1 - beta1)
            st["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)

            # (optional) bias-correction – cheap, so keep it
            step = st.get("t", 0) + 1
            st["t"] = step
            m_hat = st["m"] / (1 - beta1 ** step)
            v_hat = st["v"] / (1 - beta2 ** step)

            # importance score
            if alpha == 1.0:                         # signal-to-noise
                #score = m_hat / (v_hat.sqrt() + eps)
                score = m_hat.abs() * v_hat
            elif alpha == 0.0:                       # |g|
                score = m_hat
            else:                                    # blend
                score = (m_hat ** alpha) * (v_hat.sqrt() ** (1 - alpha))

            flat_scores.append(score.flatten())

        return torch.cat(flat_scores)

    def _moment_update(self, params, grads, *, beta1=0.9, beta2=0.999):
        """
        Update running |grad| (m) and grad² (v) for every parameter.
        Nothing is returned; self._mom_state is modified in-place.
        """
        if not hasattr(self, "_mom_state"):
            self._mom_state = {}

        for p, g in zip(params, grads):
            if g is None:
                g = torch.zeros_like(p)

            key = (id(p), p.shape)
            st  = self._mom_state.get(key)
            if st is None or st["m"].shape != p.shape:
                st = {"m": torch.zeros_like(p),
                    "v": torch.zeros_like(p),
                    "t": 0}
                self._mom_state[key] = st

            st["m"].mul_(beta1).add_(g.abs(), alpha=1 - beta1)
            st["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            st["t"] += 1

