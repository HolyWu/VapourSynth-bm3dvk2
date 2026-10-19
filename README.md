# bm3dvk2

Block-Matching and 3D filtering, using Vulkan compute.

Ported from [VapourSynth-BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D). BM3D (spatial) and V-BM3D (temporal), basic and final estimates are consolidated into a single function. Temporal results are aggregated internally, so no separate aggregation call is needed.


## Parameters

```py
bm3dvk2.BM3D(vnode clip[, vnode ref=None, str profile="fast", float[] sigma=[10, 10, 10], int radius, int block_size, int block_step, int group_size, int bm_range, int bm_step, int ps_num, int ps_range, int ps_step, float th_mse, float hard_thr, int matrix=2])
```

- clip: Clip to process. Gray or RGB of 16/32 bit float is supported, at most 65536 in width and height. RGB is converted to the opponent color space internally and converted back for output. 16 bit float input is processed with half precision arithmetic where the device supports it, which holds content within the usual [0, 1] range: samples far above it, even a single one, can overflow (from about 370-1000 with the profile defaults, down to about 20 at block_size=64 and group_size=256), so use 32 bit float for such content. Some estimates are accumulated with a fixed absolute precision of about 2^-17, so content scaled far below [0, 1] as a whole (to about 1e-4 of that range or less, with sigma scaled to match) loses accuracy; keep content in [0, 1].

- ref: Reference clip, of the same format, dimensions and length as `clip`. When omitted, the basic estimate (collaborative hard-thresholding) is produced. When specified, the final estimate (collaborative Wiener filtering) is produced, with `ref` (usually a basic estimate) guiding block-matching and the Wiener filter.

- profile: Preset of the defaults below: `"fast"`, `"lc"` (low complexity), `"np"` (normal), `"high"` or `"vn"` (very noisy).

- sigma: Strength of denoising per plane, in 8 bit scale, valid range [0, +inf). Unassigned planes take the value of the previous one. Planes of 0 are left unfiltered, and when it is 0 for every plane, the input is returned unprocessed.

- radius: Temporal radius, valid range [0, 16]. 0 denoises spatially (BM3D), greater values denoise temporally over `radius * 2 + 1` frames (V-BM3D). Defaults to the radius of the profile, and the defaults of the other parameters follow the selected mode.

- block_size: Size of a block, valid range [1, 64]. It must not exceed the width or height of `clip`, which also applies to the profile default.

- block_step: Step between reference blocks, valid range [1, block_size].

- group_size: Maximum number of blocks in a group, valid range [1, 256].

- bm_range: Radius of the block-matching search window, valid range [1, +inf).

- bm_step: Step between block-matching search positions, valid range [1, bm_range].

- ps_num: Number of the best matches of a frame that center the predictive search in the next frame, valid range [1, group_size]. Temporal only.

- ps_range: Radius of the predictive search window, valid range [1, +inf). Temporal only.

- ps_step: Step between predictive search positions, valid range [1, ps_range]. Temporal only.

- th_mse: Threshold of mean square error for block-matching, valid range [0, +inf). Defaults depend on the profile and `sigma[0]`. With `block_size` above 8, block distances carry a rounding error that grows with the square of the sample range, so for content far outside [0, 1] and a small `th_mse`, identical blocks may go unmatched.

- hard_thr: Threshold of hard-thresholding, valid range (0, +inf). Basic estimate only.

- matrix: Matrix coefficients of Gray input, used to normalize `sigma` and `th_mse`: 0 (GBR), 1 (BT.709), 2 (unspecified, chosen by resolution), 4 (FCC), 5 (BT.470BG), 6 (SMPTE 170M), 7 (SMPTE 240M), 9 (BT.2020 NCL), 10 (BT.2020 CL) or 100 (opponent color space). RGB input always uses the opponent color space. Gray input frames with the property `BM3D_OPP=1` (opponent color space planes, e.g. from `bm3d.RGB2OPP`) must use 100, or their frame requests fail.


## Profile defaults

Values are basic / final.

| profile | radius | block_size | block_step | group_size | bm_range | ps_num | ps_range | hard_thr |
|---------|--------|------------|------------|------------|----------|--------|----------|----------|
| radius = 0 |
| fast | | 8 / 8 | 8 / 7 | 8 / 8 | 9 / 9 | | | 2.7 |
| lc | | 8 / 8 | 6 / 5 | 16 / 16 | 9 / 9 | | | 2.7 |
| np | | 8 / 8 | 4 / 3 | 16 / 32 | 16 / 16 | | | 2.7 |
| high | | 8 / 8 | 3 / 2 | 16 / 32 | 16 / 16 | | | 2.7 |
| vn | | 8 / 11 | 4 / 6 | 32 / 32 | 16 / 16 | | | 2.8 |
| radius > 0 |
| fast | 1 | 8 / 8 | 8 / 7 | 8 / 8 | 7 / 7 | 2 / 2 | 4 / 5 | 2.7 |
| lc | 2 | 8 / 8 | 6 / 5 | 8 / 8 | 9 / 9 | 2 / 2 | 4 / 5 | 2.7 |
| np | 3 | 8 / 8 | 4 / 3 | 8 / 8 | 12 / 12 | 2 / 2 | 5 / 6 | 2.7 |
| high | 4 | 8 / 8 | 3 / 2 | 8 / 8 | 16 / 16 | 2 / 2 | 7 / 8 | 2.7 |
| vn | 4 | 8 / 11 | 4 / 6 | 16 / 16 | 12 / 12 | 2 / 2 | 5 / 6 | 2.8 |

`bm_step` and `ps_step` default to 1. `th_mse` defaults to `sigma[0] * 80 + 400` / `sigma[0] * 10 + 200`, or `sigma[0] * 150 + 1000` / `sigma[0] * 40 + 400` for `"vn"`.


## Installation

```
pip install -U vapoursynth-bm3dvk2
```
