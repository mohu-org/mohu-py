# MOHU Python College Demo

## Fresh run

    cd ~/Projects/mohu-org/mohu-py
    python3 -m venv .venv-demo
    source .venv-demo/bin/activate
    python -m pip install -U pip maturin pytest
    maturin develop
    pytest
    python examples/college_demo.py
    python examples/college_demo.py

The demo proves Python construction, metadata inspection, logical transpose,
reshape, readable representation, and Rust-backed 2-D matrix multiplication.
The second run is a simple live-demo safety check.

## What to say

- `mu.array` creates a Python-owned handle backed by MOHU's Rust `Buffer`.
- `shape`, `ndim`, `dtype`, `strides`, and `size` expose runtime layout metadata.
- `T` is a logical transpose view; `tolist()` follows logical order for views.
- `reshape` uses the existing Rust Buffer semantics.
- `@` reuses `mohu-ops` matrix multiplication; the 2 x 2 result is
  `[[19.0, 22.0], [43.0, 50.0]]`.

## Prototype limits

This is a deliberately narrow F64 prototype: non-empty 1-D and rectangular
2-D inputs only. It has no mutation, indexing, slicing, NumPy protocols,
additional dtypes, or inferred reshape dimensions.
