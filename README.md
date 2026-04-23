# Forget Forgetting: Continual Learning in a World of Abundant Memory

**ICLR 2026** [[paper](https://arxiv.org/abs/2502.07274)]

**Authors**: Dongkyu Cho, Taesup Moon, Rumi Chunara, Kyunghyun Cho, Sungmin Cha

> Continual learning (CL) has traditionally focused on minimizing exemplar memory, a constraint often misaligned with modern systems where GPU time, not storage, is the primary bottleneck. This paper challenges this paradigm by investigating a more realistic regime: one where memory is abundant enough to mitigate forgetting, but full retraining from scratch remains prohibitively expensive. We propose **Weight Space Consolidation (WSC)**, a lightweight method that combines (1) rank-based parameter resets to restore plasticity with (2) weight averaging to enhance stability. Validated on both class-incremental learning with image classifiers and continual instruction tuning with large language models, our approach outperforms strong baselines while matching the low computational cost of replay.

This codebase was built using [PyCIL](https://github.com/LAMDA-CL/PyCIL) using PyTorch. We thank the authors of PyCIL for their amazing continual learning codebase.


## Introduction

This codebase reproduces the class-incremental learning (class-IL) experiments from the paper. Each method is selected by the `model_name` field in a JSON config under `exps/` (CIFAR-100) or `exps_imagenet100/` (ImageNet-100) and constructed in `utils/factory.py`.


## Methods in the paper experiments

The shell drivers `run_cifar.sh` and `run_imagenet.sh` sweep memory sizes for the learners below. Unless noted, methods follow the standard PyCIL-style pipeline: train on new classes with replay from an exemplar buffer, then update the buffer for the next task.

| `model_name` | Module | Role |
| --- | --- | --- |
| `replay` | `models/replay.py` | **Replay baseline**: finetune the classifier and backbone on the union of new data and stored exemplars (cross-entropy only). |
| `icarl` | `models/icarl.py` | **iCaRL**: exemplar replay plus knowledge distillation from the previous model and nearest-class-mean classification at inference. |
| `bic` | `models/bic.py` | **BiC**: replay with a linear bias-correction layer trained on a held-out validation split to reduce classifier imbalance between old and new classes. |
| `wa` | `models/wa.py` | **WA (weight aligning)**: replay with explicit **weight alignment** of the FC layer after each task to keep magnitudes comparable across classes. |
| `der` | `models/der.py` | **DER**: dynamic expandable representation—adds parallel residual branches per task and combines them for the current class count (PyCIL-style DERNet). |
| `foster` | `models/foster.py` | **FOSTER**: feature boosting and compression with a student–teacher setup, optional weight averaging on student/teacher, and shrinkage of expanded capacity over tasks. |
| `memo` | `models/memo.py` | **MEMO**: memory-efficient design with task-agnostic “generalized” blocks and smaller task-adaptive blocks; trains which blocks to adapt per stage. |
| `wsc` | `models/wsc.py` | **WSC (ours)**: replay-based training with **rank-based parameter resets** and **stochastic weight averaging (SWA)**. After validation loss plateaus on each new task, low-importance weights (by Adam-style EMA of \|gradient\| and squared gradient) are partially blended back toward the previous task’s weights, then SWA (`AveragedModel` / `SWALR`, with BN recalibration) stabilizes the solution. Originally registered as **`replay_ours_swa`** (SWA + trim)—not `replay_ours`. |

Additional algorithms from PyCIL remain registered in `utils/factory.py` (e.g. `ewc`, `gem`, `podnet`) for reuse but are not part of the default paper sweep.


## Weight Space Consolidation (WSC) configs and naming

- **Implementation**: `models/wsc.py`, class `WSC`.
- **Config key**: set `"model_name": "wsc"` in JSON (configs live under `exps/wsc_memory/` and `exps_imagenet100/wsc_memory/`, e.g. `wsc_20.json` for 20 exemplars per class).
- **Logs**: runs are logged under `logs/wsc/<dataset>/...` (same pattern as other methods).
- **Backward compatibility**: `utils/factory.py` still accepts the legacy name **`replay_ours_swa`** (the original implementation string) and maps it to the same `WSC` class as `wsc`. The identifier **`replay_ours` is not** this method and is not supported as an alias (it will raise a clear error if used).

### LLM experiments (TRACE benchmark)

Continual **instruction tuning** with WSC is evaluated on **[TRACE](https://github.com/BeyonderXX/TRACE)** (DeepSpeed + Hugging Face), not the PyCIL loop above. TRACE itself is **not** vendored here; instead this repo provides a **small bridge package** under [`trace/`](trace/) so you can reuse the same WSC ideas (moment-style importance scores, pre-SWA trimming, SWA helpers) inside your own TRACE checkout or fork.

| Item | Location |
| --- | --- |
| Core helpers | [`trace/wsc_llm.py`](trace/wsc_llm.py) — portable primitives for LLM WSC (import from a copied path or extend in JAX as needed). |
| Installer | [`trace/install_into_trace.py`](trace/install_into_trace.py) — copies `wsc_llm.py` into a TRACE tree and writes `training/WSC_FROM_FORGET_FORGETTING.md` with import hints. |
| Integration guide | [`trace/INTEGRATION.md`](trace/INTEGRATION.md) — wiring `wsc_llm` into `training/*.py`, replay, and optional `--CL_method` registration. |
| Overview | [`trace/README.md`](trace/README.md) — how TRACE registers methods and where WSC fits. |

**Install into a local TRACE clone** (from this repository’s `src` directory):

```bash
python trace/install_into_trace.py --trace-root /path/to/TRACE
```

Use `--dry-run` to preview, `--force` to overwrite an existing `wsc_llm.py`. After installation, follow `trace/INTEGRATION.md` and your continual DeepSpeed script’s task loop (epochs, replay ratio, SWA schedule) as in the upstream benchmark.


## Citation

If you use this code or the **Weight Space Consolidation** method, please cite:

```bibtex
@misc{cho2026forgetforgettingcontinuallearning,
      title={Forget Forgetting: Continual Learning in a World of Abundant Memory},
      author={Dongkyu Cho and Taesup Moon and Rumi Chunara and Kyunghyun Cho and Sungmin Cha},
      year={2026},
      eprint={2502.07274},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.07274},
}
```


## Methods Tested (bibliography)

- `Replay`: exemplar replay baseline.
- `iCaRL`: Incremental Classifier and Representation Learning, CVPR 2017 [[paper](https://arxiv.org/abs/1611.07725)]
- `BiC`: Large Scale Incremental Learning, CVPR 2019 [[paper](https://arxiv.org/abs/1905.13260)]
- `WA`: Maintaining Discrimination and Fairness in Class Incremental Learning, CVPR 2020 [[paper](https://arxiv.org/abs/1911.07053)]
- `DER`: Dynamically Expandable Representation for Class Incremental Learning, CVPR 2021 [[paper](https://arxiv.org/abs/2103.16788)]
- `FOSTER`: Feature Boosting and Compression for Class-incremental Learning, ECCV 2022 [[paper](https://arxiv.org/abs/2204.04662)]
- `MEMO`: A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning, ICLR 2023 Spotlight [[paper](https://openreview.net/forum?id=S07feAlQHgM)]
- `WSC`: Weight Space Consolidation (this paper) [[paper](https://arxiv.org/abs/2502.07274)]


## How To Use

### Data
1. CIFAR-100: The CIFAR-100 dataset will be automatically downloaded when the CIFAR-100 experiments are run.
2. ImageNet-100: ImageNet-100 can be easily obtained on the web. In our case, we used [Kaggle](https://www.kaggle.com/datasets/ambityga/imagenet100).
- If you face an error in loading the data into the loader, please check the folder locations. Also have a look at the data pre-processing code for ImageNet, which will automatically split the data for you (`./notebooks/imagenet_data_preparation.ipynb`).

Use `./notebooks/cifar100_visualization.ipynb` to aggregate and plot log files under `logs/`.


### Dependencies (same as PyCIL)

1. [torch 1.8.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.6.0](https://github.com/pytorch/vision)
3. [tqdm](https://github.com/tqdm/tqdm)
4. [numpy](https://github.com/numpy/numpy)
5. [scipy](https://github.com/scipy/scipy)
6. [quadprog](https://github.com/quadprog/quadprog)
7. [POT](https://github.com/PythonOT/POT)

### Run experiment

1. To reproduce the CIFAR-100 experiments:
```bash
sh run_cifar.sh
```

2. To reproduce the ImageNet-100 experiments:
```bash
sh run_imagenet.sh
```

3. To run a single WSC job (example):
```bash
python main.py --config=./exps/wsc_memory/wsc_20.json
```


## Acknowledgments

We follow the practice of the original repository [PyCIL](https://github.com/LAMDA-CL/PyCIL) and thank the following repos for providing helpful components in our work.

- [Continual-Learning-Reproduce](https://github.com/zhchuu/continual-learning-reproduce)
- [GEM](https://github.com/hursung1/GradientEpisodicMemory)
- [FACIL](https://github.com/mmasana/FACIL)
- [PyCIL](https://github.com/LAMDA-CL/PyCIL)
- [TRACE](https://github.com/BeyonderXX/TRACE)
