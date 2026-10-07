# CusADi — fork with code-generation speedups

This is a fork of [CusADi](https://github.com/se-hwan/cusadi), a framework for parallelizing CasADi functions on the GPU. For the project description, installation and usage, see the [upstream repository](https://github.com/se-hwan/cusadi). Everything described there applies unchanged to this fork.

This fork adds two fixes and three performance changes. They were made for the computational studies of the time-decomposed LISCO paper ([lisco-decompose](https://github.com/lukaslueken/lisco-decompose)), where CusADi evaluates KKT residuals and their gradients for large batches during training.

If you use CusADi, please cite the original work:

```
@misc{jeon2024cusadigpuparallelizationframework,
      title={CusADi: A GPU Parallelization Framework for Symbolic Expressions and Optimal Control},
      author={Se Hwan Jeon and Seungwoo Hong and Ho Jae Lee and Charles Khazoom and Sangbae Kim},
      year={2024},
      eprint={2408.09662},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2408.09662},
}
```

## Changes

### Fixes

1. **Package-relative import** (`src/CusadiFunction.py`). `from src import CUSADI_BUILD_DIR` is now `from . import CUSADI_BUILD_DIR`. The absolute import resolved to the *calling* project's `src` package whenever CusADi was used inside a project that has its own `src/` directory, and failed there.
2. **Exact function-name match** (`run_codegen.py`). `--fn=<name>` now selects the `.casadi` file whose name equals `<name>`. Previously every file whose path *contained* `<name>` was selected, so e.g. `--fn=f_b1` also compiled `f_b1024`.

### Performance

Change 3 affects the kernels of both `generateCUDACodeDouble` (used by `run_codegen.py` by default) and `generateCUDACodeFloat`, since both use the same operation templates. Change 4 affects only `generateCUDACodeDouble`.

3. **Structure-of-arrays work array** (`src/CusadiOperations.py`). All templates in `OP_CUDA_DICT` now address the work array as `work[k * batch_size + idx]` instead of `work[idx * n_w + k]`.
   - The generated kernels run one thread per function instance, and every intermediate value lives in the global `work` array. With the previous array-of-structs layout, neighbouring threads accessed addresses `n_w * 8` bytes apart, so memory accesses were not coalesced. The kernels were bound by memory transactions, not by FP64 arithmetic.
   - The host side is unchanged: the same flat buffer is allocated and interpreted transposed.
4. **Asynchronous kernel launch** (`src/generateCUDACode.py`). The generated `evaluate()` wrapper no longer creates CUDA timing events or synchronizes the device on every call.
   - `generateCUDACodeDouble` now defaults to `benchmarking=False, debug_mode=False`.
   - `benchmarking=True` restores the timing, and now destroys its events; `debug_mode=True` restores the error check and device synchronization.
   - `evaluate()` keeps its `float` return type and returns `0` when timing is off.
   - The kernel runs on the CUDA default stream. PyTorch operations on the default stream, which PyTorch uses unless told otherwise, are ordered after it, so no explicit synchronization is needed. If CusADi is called from a non-default PyTorch stream or during CUDA-graph capture, use `debug_mode=True`, which synchronizes after each launch.
   - Kernel launch errors are no longer checked right away. They surface at the next synchronizing CUDA call.
5. **No per-call buffer zeroing** (`src/CusadiFunction.py`). `evaluate()` no longer zeroes the work and output tensors before each call. The generated kernel writes every work slot before reading it and writes every output nonzero, so both buffers are fully overwritten anyway. This was verified by filling all buffers with NaN before evaluating: no NaN reached any output. Set the environment variable `CUSADI_CLEAR_TENSORS=1` before importing `cusadi` to restore the zeroing; it is read once at import.

### Effect

Measured on an RTX 4090 for a learned-solver training step (batch size 1024, NMPC horizon 100):

| | before | after | speedup |
|---|---|---|---|
| CusADi kernels per step (change 3) | 34.0 ms | 8.2 ms | 4.1× |
| full training step (change 3) | 75.0 ms | 51.0 ms | 1.47× |
| full training step (changes 3–5) | 75.0 ms | 48.2 ms | 1.56× |

The gain depends on problem and batch size. For small batches, where the GPU is far from saturated, it can be negligible.

**Numerics.** Outputs agree with upstream CusADi to within 1 ULP (e.g. 2.2e-16 relative). The kernels perform the same operations in the same per-thread order. The remaining differences come from different fused-multiply-add contractions by `nvcc` after the index arithmetic changed.

**Rebuild after switching.** Libraries in `build/` that were generated with upstream CusADi are not regenerated automatically. Rerun `python run_codegen.py --fn=<name>` for existing functions to get the new kernels.

## License

MIT, as upstream. See `LICENSE.txt` (original copyright notice retained).
