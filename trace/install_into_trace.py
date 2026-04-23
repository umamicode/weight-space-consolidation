#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _src_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_trace_layout(trace_root: Path, dest_name: str) -> Path:
    dest = trace_root / dest_name
    if not trace_root.is_dir():
        raise SystemExit(f"TRACE root is not a directory: {trace_root}")
    if not dest.is_dir():
        raise SystemExit(
            f"Expected ``{dest_name}/`` under TRACE root (clone https://github.com/BeyonderXX/TRACE). Missing: {dest}"
        )
    marker = trace_root / "README.md"
    if not marker.exists():
        sys.stderr.write(f"Warning: no README.md at {trace_root} — is this a TRACE clone?\n")
    return dest


def _write_instructions(trace_root: Path, dest_dir: Path, dry_run: bool) -> None:
    note = dest_dir / "WSC_FROM_FORGET_FORGETTING.md"
    body = f"""# WSC files added by Forget Forgetting / WSC_SRC

Installed (UTC): {datetime.now(timezone.utc).isoformat()}

## Files

- **`wsc_llm.py`** — moment trim + DeepSpeed SWA helpers (`pre_swa_trim`, `make_swa`, …).

## Edit your continual LLM training script under ``{dest_dir.name}/``

1. Remove duplicate definitions of ``pre_swa_trim``, ``make_swa``, and any local moment-score helpers if present.
2. Add imports (after ``torch`` / DeepSpeed imports is fine):

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

3. Replace manual CPU snapshots with ``snapshot_trainable_params_cpu(model)`` where you build ``prev_state``.
4. After each task, call ``clear_moment_score_state()`` if you reuse the global moment tracker.
5. For plain PyTorch tests only, ``from wsc_llm import make_swa_torch`` (not used in DeepSpeed TRACE runs).

Full notes: see the **Forget Forgetting** repository ``TRACE/INTEGRATION.md`` (or copy that file here next time you run the installer with ``--with-docs``).

Paper: https://arxiv.org/abs/2502.07274  
TRACE: https://github.com/BeyonderXX/TRACE
"""
    if dry_run:
        print(f"[dry-run] would write {note}")
        return
    note.write_text(body, encoding="utf-8")
    print(f"Wrote {note}")


def main() -> None:
    p = argparse.ArgumentParser(description="Install wsc_llm.py into a TRACE clone.")
    p.add_argument(
        "--trace-root",
        type=Path,
        required=True,
        help="Path to the root of a cloned TRACE repository.",
    )
    p.add_argument(
        "--dest",
        type=str,
        default="training",
        help="Subdirectory under TRACE root (default: training).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned copies only.")
    p.add_argument("--force", action="store_true", help="Overwrite existing wsc_llm.py.")
    p.add_argument(
        "--with-docs",
        action="store_true",
        help="Also copy INTEGRATION.md to TRACE root as WSC_INTEGRATION.md.",
    )
    args = p.parse_args()

    trace_root = args.trace_root.expanduser().resolve()
    dest_dir = _ensure_trace_layout(trace_root, args.dest)

    here = Path(__file__).resolve().parent
    src_file = here / "wsc_llm.py"
    if not src_file.is_file():
        raise SystemExit(f"Missing vendor file: {src_file}")

    dst_file = dest_dir / "wsc_llm.py"
    if dst_file.exists() and not args.force:
        if args.dry_run:
            print(f"[dry-run] target exists (would need --force to replace): {dst_file}")
        else:
            raise SystemExit(
                f"Refusing to overwrite {dst_file} (pass --force to replace). "
                "Backup the file or remove it first."
            )

    if args.dry_run:
        print(f"[dry-run] would copy:\n  {src_file}\n  -> {dst_file}")
    else:
        shutil.copy2(src_file, dst_file)
        print(f"Copied:\n  {src_file}\n  -> {dst_file}")

    if args.with_docs:
        doc_src = here / "INTEGRATION.md"
        doc_dst = trace_root / "WSC_INTEGRATION.md"
        if not doc_src.is_file():
            sys.stderr.write(f"Warning: missing {doc_src}, skip --with-docs\n")
        elif args.dry_run:
            print(f"[dry-run] would copy:\n  {doc_src}\n  -> {doc_dst}")
        else:
            shutil.copy2(doc_src, doc_dst)
            print(f"Copied:\n  {doc_src}\n  -> {doc_dst}")

    _write_instructions(trace_root, dest_dir, args.dry_run)

    if not args.dry_run:
        print(
            "\nNext: edit your continual LLM training script under "
            f"``{dest_dir.name}/`` in the TRACE clone to import from ``wsc_llm``; see "
            f"{dest_dir / 'WSC_FROM_FORGET_FORGETTING.md'}"
        )


if __name__ == "__main__":
    if __package__ is None:
        root = _src_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    main()
