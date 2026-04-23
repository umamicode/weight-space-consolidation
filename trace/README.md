# TRACE and Weight Space Consolidation (WSC)

This folder documents how **[TRACE](https://github.com/BeyonderXX/TRACE)**—the continual learning benchmark for large language models from [Wang et al. (TRACE, arXiv:2310.06762)](https://arxiv.org/abs/2310.06762)—relates to **Weight Space Consolidation (WSC)** from *[Forget Forgetting: Continual Learning in a World of Abundant Memory](https://arxiv.org/abs/2502.07274)*.

## Contents of this directory

| File | Purpose |
| --- | --- |
| **`wsc_llm.py`** | Portable **LLM WSC** primitives: moment scoring on LM loss, `pre_swa_trim`, DeepSpeed `make_swa` / `SWALR`, SWA weight swap, optional BN helper, and `make_swa_torch` for plain PyTorch optimizers. |
| **`install_into_trace.py`** | **Copies** `wsc_llm.py` into a local TRACE clone (`--trace-root /path/to/TRACE`) and writes `training/WSC_FROM_FORGET_FORGETTING.md` with import hints. Run from `src`: `python TRACE/install_into_trace.py --trace-root ...` |
| **`INTEGRATION.md`** | Step-by-step: install or `PYTHONPATH` this module into a **TRACE** clone and import it from your continual LLM training script under `training/` (or register `--CL_method WSC`). |
| **`README.md`** (this file) | Benchmark overview, `Method2Class` / `--CL_method` notes, and links upstream. |

**Vision (CIFAR / ImageNet) WSC** remains in the parent repo: `models/wsc.py` with `model_name: "wsc"`.

TRACE itself is **not** vendored here. Clone the official repository separately:

```bash
git clone https://github.com/BeyonderXX/TRACE.git
cd TRACE
pip install -r requirements.txt
```

Upstream README (datasets, `data_path`, replay ratio, inference paths): [github.com/BeyonderXX/TRACE](https://github.com/BeyonderXX/TRACE).

**→ To add WSC into a TRACE checkout:** run **`python TRACE/install_into_trace.py --trace-root /path/to/TRACE`** from `src`, then read [`INTEGRATION.md`](INTEGRATION.md) and wire your continual LLM training script to `from wsc_llm import ...`.


## What TRACE provides

From the [TRACE README](https://github.com/BeyonderXX/TRACE):

- **Benchmark**: nine task streams (e.g. C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE variants, 20Minuten) plus a replay anchor dataset (e.g. Lima), with a prescribed prompt/answer JSON format.
- **Training**: DeepSpeed + Hugging Face causal LMs; scripts for naive SFT, LoRA, replay, and **continual learning methods**.
- **Key CLI flag**: `--CL_method` selects which continual-learning wrapper is used when you drive training through `training/main.py` (see below).

Data layout, collators, and evaluation flows are described in the upstream repo; this file only explains **where WSC lives** and **how to register a method** in TRACE-style code.


## How TRACE registers “methods” (`--CL_method`)

Continual methods that plug into **`training/main.py`** are looked up from **`training/params.py`**, which defines a dictionary roughly of the form:

```python
Method2Class = {
    "PP": PP,
    "EWC": EWC,
    "GEM": GEM,
    "OGD": OGD,
    "LwF": LwF,
    "L2P": L2P,
    "MbPA++": MbPAplusplus,
    "LFPT5": LFPT5,
    "O-LoRA": O_LoRA,
    "base": CL_Base_Model,
    "lora": lora,
}
```

After the model is wrapped (if needed), DeepSpeed is initialized, and **`main.py`** does:

```python
if args.CL_method in Method2Class.keys():
    CL_Trainer = Method2Class[args.CL_method](
        model, tokenizer, optimizer,
        train_task_list, eval_task_list, test_task_list, args
    )
    CL_Trainer.train_continual()
```

So any new method you want in **that** list must:

1. **Implement a trainer class** with the same constructor signature as the existing entries (see e.g. `model/Dynamic_network/PP.py`: class `PP(CL_Base_Model)` and its `__init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args, ...)`).
2. **Expose** `train_continual(self)` (and any hooks TRACE expects—mirror `CL_Base_Model` subclasses such as `EWC`, `GEM`, or `PP`).
3. **Register** the class in `training/params.py`: add an import and a new key, e.g. `"WSC": WSC_Trainer`.
4. **Extend `training/main.py` if needed**: several methods require **extra branches before** `deepspeed.initialize` (special model surgery for `PP` / `L2P`, prompts for `LFPT5`, etc.). If WSC needs similar setup, add a guarded block `if args.CL_method == "WSC": ...` next to those.
5. **Wire shell scripts**: duplicate an existing `scripts/train_seq_cl.sh`-style script and pass `--CL_method WSC` (and document it next to replay / LoRA in the upstream README if you contribute back).

If you only add the dict entry without a matching `train_continual` implementation, training will fail at instantiation or the first step—match the interface of an existing method first, then specialize the inner loop.


## Where WSC lives in a TRACE-style checkout

For LLM continual tuning, **Weight Space Consolidation** is normally a **standalone DeepSpeed training script** under `training/` (moment trim + SWA on top of replay), **not** a `Method2Class` key. You keep that script in your fork (or start from TRACE replay-style scripts) and import **`wsc_llm`** for the shared math.

Two coherent ways to ship WSC in TRACE:

| Goal | What to do |
| --- | --- |
| **Standalone script** | Clone [TRACE](https://github.com/BeyonderXX/TRACE), run **`install_into_trace.py`**, then edit your continual LLM script under `training/` to import `wsc_llm` and add a `deepspeed` shell script under `scripts/`. |
| **`--CL_method WSC`** | Implement a `CL_Base_Model` subclass with the same schedule as your standalone script, register it in `params.py`, extend `main.py` only if needed, and add a shell script that passes `--CL_method WSC`. |

The vision-classifier WSC implementation in **this** repository remains under `models/wsc.py` and `model_name: "wsc"` in PyCIL configs; the TRACE tree is the right place for the **LLM** DeepSpeed entrypoint.


## License and attribution

- **TRACE** is released under **Apache-2.0** ([LICENSE](https://github.com/BeyonderXX/TRACE/blob/master/LICENSE) in the upstream repo). If you copy TRACE files into another project, keep the original headers and license notices.
- Cite TRACE when you use the benchmark; cite [Forget Forgetting / WSC](https://arxiv.org/abs/2502.07274) when you report the consolidation method.


## Quick checklist (add WSC to `Method2Class`)

1. In your TRACE clone, add `model/.../wsc.py` (name arbitrary) defining `class WSC(CL_Base_Model): ...` with `train_continual` implementing trim + SWA + replay to match your validated standalone script.
2. Edit `training/params.py`: `from model....wsc import WSC` and `"WSC": WSC` inside `Method2Class`.
3. Edit `training/main.py` only if WSC needs preprocessing like `PP`/`L2P` (optional `if args.CL_method == "WSC":` block before `deepspeed.initialize`).
4. Add `scripts/.../train_wsc.sh` mirroring other methods, passing `--CL_method WSC`.
5. Update the upstream [README](https://github.com/BeyonderXX/TRACE) method bullet list (`CL_method` …) so others see **WSC** next to EWC, GEM, etc., or open a PR to TRACE with the same text.

For questions about the benchmark itself, use the [TRACE repository](https://github.com/BeyonderXX/TRACE) issues and documentation.
