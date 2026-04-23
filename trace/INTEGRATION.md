# Integrating WSC (`wsc_llm.py`) into the official TRACE repository

This document explains how to use **`wsc_llm.py`** inside a clone of **[TRACE](https://github.com/BeyonderXX/TRACE)** so Weight Space Consolidation stays aligned with the [Forget Forgetting](https://arxiv.org/abs/2502.07274) LLM recipe (moment trim + DeepSpeed SWA) without copying a whole private research tree.


## Step 1 — Clone TRACE

```bash
git clone https://github.com/BeyonderXX/TRACE.git
cd TRACE
pip install -r requirements.txt
```


## Step 2 — Add `wsc_llm.py` to the TRACE tree

### Automated (recommended)

From the **WSC_SRC `src`** directory, after cloning TRACE:

```bash
python TRACE/install_into_trace.py --trace-root /path/to/TRACE
```

Optional:

- `--force` — replace an existing `training/wsc_llm.py`.
- `--with-docs` — also copy `INTEGRATION.md` to `WSC_INTEGRATION.md` at the TRACE repo root.
- `--dry-run` — show what would happen without writing files.

The installer copies `wsc_llm.py` into `TRACE/training/wsc_llm.py` and writes `training/WSC_FROM_FORGET_FORGETTING.md` with import snippets.

### Manual

Pick **one** location (both work with `PYTHONPATH=.` from the TRACE repo root):

| Location | Import in training scripts |
| --- | --- |
| `TRACE/training/wsc_llm.py` | `from wsc_llm import pre_swa_trim, make_swa_deepspeed, ...` |
| `TRACE/utils/wsc_llm.py` | `from utils.wsc_llm import pre_swa_trim, ...` |

**Copy the file** from this repository:

- Source: `WSC_SRC/src/TRACE/wsc_llm.py`  
- Destination: e.g. `TRACE/training/wsc_llm.py`

Keep the **Apache-2.0 / SPDX** header at the top of `wsc_llm.py` when redistributing.

Do **not** copy the whole `WSC_SRC` tree into TRACE unless you intend to maintain a fork; one file (plus your edited training script) is enough.


## Step 3 — Wire the training script

Official TRACE uses `training/main.py` + `--CL_method` for many baselines. WSC LLM runs use a **standalone DeepSpeed script** under `training/` that owns the task loop (same pattern as replay-style scripts in upstream TRACE).

**Option A — Minimal change (recommended)**  
In your fork, open the continual LLM script that already implements WSC (or add a new one, e.g. `training/wsc_train.py`):

1. Remove duplicate local definitions of `pre_swa_trim`, `make_swa`, and any moment-score helpers that duplicate `wsc_llm`.
2. Add:

```python
from wsc_llm import (
    clear_moment_score_state,
    make_swa,
    pre_swa_trim,
    snapshot_trainable_params_cpu,
    swap_swa_weights_to_base,
    update_bn_if_applicable,
)
```

`wsc_llm.make_swa` is the DeepSpeed entrypoint (same idea as a local `make_swa` that unwraps the ZeRO optimizer).

3. Replace manual CPU snapshots with:

```python
from wsc_llm import snapshot_trainable_params_cpu

prev_state = None
if i_task >= 1:
    prev_state = snapshot_trainable_params_cpu(model.module if hasattr(model, "module") else model)
```

4. After each task, call `clear_moment_score_state()` if you reuse the global moment tracker.

5. Where you call `update_bn` on the SWA model, you may use `update_bn_if_applicable(train_dataloader, swa_model, device=device)`.

6. After SWA, swap weights:

```python
base = model.module if hasattr(model, "module") else model
swap_swa_weights_to_base(swa_model, base)
```

**Option B — `PYTHONPATH` without copying**  
From the TRACE repo root:

```bash
export PYTHONPATH="/path/to/WSC_SRC/src:${PYTHONPATH}"
```

Then:

```python
from TRACE.wsc_llm import pre_swa_trim, make_swa_deepspeed as make_swa, ...
```

This keeps a single source of truth in `WSC_SRC` but ties your TRACE checkout to that path (fragile for clusters / collaborators).


## Step 4 — Shell scripts

Add a `deepspeed` launch script under `scripts/` that points at your patched continual training script (mirror the layout of existing `scripts/train_replay.sh`-style entries). No change to `wsc_llm.py` is required if the Python API matches.


## Step 5 — (Optional) Expose `--CL_method WSC` in `main.py`

1. Implement `class WSC(CL_Base_Model)` with `train_continual()` that reproduces the same schedule as your standalone script (task data + replay + trim + SWA).
2. Register in `training/params.py` → `Method2Class["WSC"] = WSC`.
3. Add `if args.CL_method == "WSC":` branches in `training/main.py` only if you need model surgery before `deepspeed.initialize`.
4. Add `scripts/.../train_wsc.sh` with `--CL_method WSC`.

Using **`wsc_llm.py` inside that class** keeps the trim/SWA math in one place.


## Verification

Run one short task on a tiny model or subset of data; confirm loss decreases and checkpoints save. Compare loss curves to a known-good run if you have one.


## License

TRACE and the original DeepSpeed training templates are **Apache-2.0**. This `wsc_llm.py` retains compatible headers; cite TRACE and [Forget Forgetting / WSC](https://arxiv.org/abs/2502.07274) in publications.
