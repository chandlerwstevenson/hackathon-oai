# GPU SRS Channel Estimator (`srs_est_cuda`)

A CUDA replacement for OAI's fixed-point SRS channel estimator. The gNB loads it with:

```bash
--loader.srs_est.shlibversion _cuda
```

For every SRS, the gNB calls `srs_est_run()`. If it returns `0`, the gNB uses our float estimate for timing advance, SNR, SRS.indication, MAC and the ISAC export. If it returns `-1`, the gNB falls back to its fixed-point estimator for that SRS.

## How it works

The estimator works in the **delay domain**. A real channel only spans a few delay taps, while noise is spread across all of them. Keeping just the taps where the channel lives removes most of the noise, and it also separates UE ports that share the same subcarriers.

```
y (comb REs) ──► LS + symbol average ──► IFFT (M) ──► window per port ──► zero-pad ──► FFT (K·M) ──► h_full
                                              │                                                       └─► h_comb (every K-th)
                                              └──► noise_var, timing (CPU)
```

| Step | Where | What it does |
|---|---|---|
| 1. Least squares | `ls_kernel` | Multiplies each comb RE by `conj(rbar)` to remove the pilot, and averages over SRS symbols. One row per (comb group, antenna). |
| 2. To delay domain | cuFFT, size `M` | Ports on the same comb carry different cyclic shifts. After the IFFT, port `p` sits at bin `n_cs·M/n_cs_max`, so the ports no longer overlap. |
| 3. Window | `window_kernel` | For each (port, antenna), keeps the taps from `win_pre` before to `win_post` after the port's position, moves them back to delay 0 and zeroes everything else. This is the denoising step. |
| 4. Back to frequency | cuFFT, size `K_TC·M` | Zero-padding to `K_TC·M` interpolates onto every subcarrier, giving `h_full`. `h_comb` is every `K_TC`-th value, taken with a strided copy. |
| 5. Reductions | host | `noise_var` from the bins outside every window, `timing_offset_s` from the strongest in-window bin, `signal_power` as the mean of `|h_comb|²`. |

The estimate keeps the channel's true timing. The detected timing is reported in `timing_offset_s` and is not removed from the estimate.

## Code layout

- **`srs_est_init()`** allocates all device buffers once for the largest case (`max_rx × max_ports × max_K × max_M`), creates the CUDA stream, and builds the FFT plans for the configured size.
- **`get_plans()`** keeps a small cache of cuFFT plan pairs (inverse size `M`, forward size `K_TC·M`), one per `(M, K_TC)`. A new SRS size builds its plans once, and later calls reuse them.
- **`srs_est_run()`** runs steps 1–5 on a single stream and synchronizes once before the CPU reads the results.
- **`srs_est_shutdown()`** waits for the stream, destroys the plans, frees the buffers and destroys the stream.

`srs_est_run()` returns `-1` (fallback) when the input is out of range, the plan cache is full, any CUDA call fails, or the port windows would overlap (`win_pre + win_post + 1` larger than the spacing between ports).

## Why it should beat fixed point

| | Fixed point (OAI) | This module |
|---|---|---|
| Port separation | Averages neighbouring REs | Separate delay window per port |
| Denoising | Incidental, a few dB | Keeps W of M taps, about 10·log₁₀(M/W) dB |
| Interpolation | Short linear FIR | Zero-padded FFT |
| Arithmetic | 16-bit fixed point | 32-bit float |

## Build and test

Build from `cmake_targets` with `CMAKE_CUDA_ARCHITECTURES` set to your GPU (`90` for GH200, `80` for A100, `121` for GB10):

```bash
cmake --build ran_build/build --target srs_est_cuda test_srs_est
cd ran_build/build

./test_srs_est                                   # unit test, synthetic channel

# fixed-point baseline, then this module, with the same settings
./nr_srssim -R 273 -z 32 -y 4 -g A,l,0 -s 0 -S 40 -N 0 -n 3
./nr_srssim -R 273 -z 32 -y 4 -g A,l,0 -s 0 -S 40 -N 0 -n 3 --loader.srs_est.shlibversion _cuda

ctest -R nr_srssim.module --output-on-failure    # the five gated scenarios
```
