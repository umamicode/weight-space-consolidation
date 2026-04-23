# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map

PyTree = TypeVar("PyTree")


def snapshot_params(params: PyTree) -> PyTree:
    return tree_map(lambda x: jnp.array(x, copy=True), params)


def init_moment_state(params: PyTree) -> Dict[str, PyTree]:
    return {
        "m": tree_map(jnp.zeros_like, params),
        "v": tree_map(jnp.zeros_like, params),
        "t": jnp.asarray(0, dtype=jnp.int32),
    }


def clear_moment_state() -> Dict[str, PyTree]:
    return {}


def compute_moment_scores_pytree(
    loss_fn: Callable[[PyTree, Any], jnp.ndarray],
    params: PyTree,
    batch: Any,
    state: Dict[str, PyTree],
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> Tuple[PyTree, Dict[str, PyTree]]:
    _, grads = jax.value_and_grad(loss_fn)(params, batch)

    if not state or "m" not in state:
        m0 = tree_map(jnp.zeros_like, params)
        v0 = tree_map(jnp.zeros_like, params)
        t_prev = jnp.asarray(0, dtype=jnp.int32)
    else:
        m0 = state["m"]
        v0 = state["v"]
        t_prev = state["t"]

    t_new = t_prev + jnp.asarray(1, dtype=jnp.int32)
    t_f = jnp.maximum(t_new.astype(jnp.float32), 1.0)

    m_new = tree_map(
        lambda m, g: beta1 * m + (1.0 - beta1) * jnp.abs(g),
        m0,
        grads,
    )
    v_new = tree_map(
        lambda v, g: beta2 * v + (1.0 - beta2) * (g * g),
        v0,
        grads,
    )

    def _score(m, v):
        m_hat = m / (1.0 - jnp.float32(beta1) ** t_f)
        v_hat = v / (1.0 - jnp.float32(beta2) ** t_f)
        return jnp.abs(m_hat) * v_hat

    scores = tree_map(_score, m_new, v_new)
    new_state = {"m": m_new, "v": v_new, "t": t_new}
    return scores, new_state


def compute_moment_flat_scores(
    loss_fn: Callable[[PyTree, Any], jnp.ndarray],
    params: PyTree,
    batch: Any,
    state: Dict[str, PyTree],
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> Tuple[jnp.ndarray, Dict[str, PyTree]]:
    scores, new_state = compute_moment_scores_pytree(
        loss_fn, params, batch, state, beta1=beta1, beta2=beta2
    )
    flat = jnp.concatenate([jnp.ravel(x) for x in tree_leaves(scores)])
    return flat, new_state


def _topk_threshold(flat: jnp.ndarray, k: int) -> jnp.ndarray:
    k = min(int(k), int(flat.size))
    vals = jax.lax.top_k(flat, k)[0]
    return jnp.min(vals)


def _trim_leaf(p: jnp.ndarray, old: jnp.ndarray, s: jnp.ndarray, keep: float) -> jnp.ndarray:
    n = int(s.size)
    k = max(int(n * keep / 100.0), 1)
    flat_s = jnp.ravel(s)
    thresh = _topk_threshold(flat_s, k)
    mask = jnp.reshape(flat_s >= thresh, s.shape)
    blend = 0.5 * p + 0.5 * old
    return jnp.where(mask, p, blend)


def pre_swa_trim(
    params: PyTree,
    prev_state: Optional[PyTree],
    scores: PyTree,
    keep: float = 20.0,
) -> PyTree:
    if prev_state is None:
        return params
    return tree_map(lambda p, o, s: _trim_leaf(p, o, s, keep), params, prev_state, scores)


def pre_swa_trim_from_batch(
    loss_fn: Callable[[PyTree, Any], jnp.ndarray],
    params: PyTree,
    batch: Any,
    prev_state: Optional[PyTree],
    moment_state: Dict[str, PyTree],
    keep: float = 20.0,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> Tuple[PyTree, Dict[str, PyTree]]:
    scores, new_moment = compute_moment_scores_pytree(
        loss_fn, params, batch, moment_state, beta1=beta1, beta2=beta2
    )
    new_params = pre_swa_trim(params, prev_state, scores, keep=keep)
    return new_params, new_moment


def swa_init(params: PyTree) -> PyTree:
    return snapshot_params(params)


def swa_update(
    swa_params: PyTree, params: PyTree, count: jnp.ndarray
) -> Tuple[PyTree, jnp.ndarray]:
    c = count.astype(jnp.float32)
    n_next = count + 1
    c_next = n_next.astype(jnp.float32)
    new_swa = tree_map(
        lambda s, p: (s * c + p) / jnp.maximum(c_next, 1.0),
        swa_params,
        params,
    )
    return new_swa, n_next


def merge_swa_into_params(swa_params: PyTree) -> PyTree:
    return snapshot_params(swa_params)


def make_swa(params: PyTree) -> Tuple[PyTree, jnp.ndarray]:
    return swa_init(params), jnp.asarray(0, dtype=jnp.int32)


def make_swa_deepspeed(*_args: Any, **_kwargs: Any) -> None:
    raise NotImplementedError("DeepSpeed SWA is PyTorch-only; use make_swa and swa_update in JAX.")


def make_swa_torch(*_args: Any, **_kwargs: Any) -> None:
    raise NotImplementedError("torch SWA is PyTorch-only; use make_swa and swa_update in JAX.")


def swap_swa_weights_to_base(swa_params: PyTree, _base_unused: Any = None) -> PyTree:
    return merge_swa_into_params(swa_params)


def update_bn_if_applicable(*_args: Any, **_kwargs: Any) -> None:
    return None
