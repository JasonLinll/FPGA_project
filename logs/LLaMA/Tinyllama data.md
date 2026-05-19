# TinyLlama-1.1B-Chat-v1.0 - Calibration 數據附錄

⚠ **此檔為 TinyLlama-specific 實驗數據, RTL 不寫死, deployment 時 calibration flow 重新生成**.

## Architecture

- Layers: 22
- Q heads: 32, KV heads: 4 (GQA ratio 8)
- head_dim: 64, num_pairs: 32

## Calibrated config (HellaSwag full 10042 verified)

| 參數 | 值 |
|------|---|
| active_dim ratio | 37.5% (24 of 64) |
| n_Q anchor | 2 |
| n_K anchor | 1 |
| anchor total | 3 (⚠ non-HW-aligned, 應改為 2 或 4) |
| β | 1.0 |
| P_THRES (mean method, deprecated) | 0.05 |
| P_THRES (per-row binary search, **TBD**) | 待 re-measure |
| block_size | 16 |
| T_margin | 0.75 |
| T_overflow | 0.25 |
| M_local | 3 |

## Accuracy (HellaSwag full 10042, acc_norm)

| Config | acc_norm | Drop |
|--------|----------|------|
| Baseline FP32 | 59.58% | - |
| RAD only | 58.46% | 1.12 pp |
| RAD + PxV (β=0.5) | 58.27% | 1.31 pp |
| RAD + PxV (β=1.0) | 58.60% | 0.98 pp |
| AWQ baseline | 58.34% | 1.24 pp |
| AWQ + RAD + PxV | 57.40% | 2.18 pp |

## MAC Saving (FP32 RAD+PxV β=1.0)

| Mechanism | Saving |
|-----------|--------|
| RAD-Cascade Q·K | 53.25% |
| PxV P·V | 45.79% |
| Total Attention | 49.52% |

High prob token: 4.1%, Low prob: 95.9%

## Cross-Task Validation

| Task | Samples | Baseline | RAD+PxV | Drop |
|------|---------|----------|---------|------|
| HellaSwag | 10042 | 59.58% | 58.27% | 1.31 pp |
| ARC-Easy | 570 | 50.18% | 49.65% | 0.53 pp |
| ARC-Challenge | 299 | 31.10% | 29.43% | 1.67 pp |
| PIQA | 1838 | 74.48% | 73.94% | 0.54 pp |
| Winogrande | 1267 | 57.30% | 55.41% | 1.89 pp |

## Context-Length (Perplexity)

| Dataset | Window | Baseline | RAD+PxV | Δ |
|---------|--------|----------|---------|---|
| WikiText-2 | 512 | 8.508 | 8.691 | +2.15% |
| PG-19 | 2048 | 11.497 | 11.993 | +4.32% |

## Active dim ablation

| Config | Drop | MAC saving |
|--------|------|------------|
| 24-dim (37.5%) | 1.12 pp | 53.25% |
| 16-dim (25.0%) | 2.87 pp | 63.93% |

## Anchor combo (2Q+1K) vs 3Q

| Layer | 3Q worst ρ | 2Q+1K worst ρ |
|-------|------------|---------------|
| 0 | 40.4% | 43.2% |
| 5 | 73.8% | 79.2% |
| 10 | 61.1% | 68.7% |
| 15 | 51.6% | 67.2% |
| 21 | 61.2% | 63.6% |

miss>64 dropped:
| Layer | 3Q | 2Q+1K |
|-------|-----|-------|
| 0 | 7.0% | 5.0% |
| 5 | 0.2% | 0.4% |
| 10 | 1.7% | 0.1% |
| 15 | 3.8% | 0.2% |
| 21 | 0.8% | 0.3% |

## Dropped dim distribution

| Layer | Kurtosis | Skewness |
|-------|----------|----------|
| 0 | 25.55 | 3.76 |
| 5 | 0.77 | 0.54 |
| 10 | 0.16 | 0.31 |
| 15 | 0.72 | 0.51 |
| 21 | 0.10 | 0.45 |

## β sweep (200 sample HellaSwag, baseline 51%)

| β | acc_norm | drop |
|---|----------|------|
| 0.3 | 50.00% | +1.00 pp |
| 0.5 | 51.00% | +0.00 pp |
| 0.7 | 50.50% | +0.50 pp |
| 1.0 | 52.00% | -1.00 pp |

## PG-19 attention prob distribution (long-context 2048)

| Layer | Per-row max mean | P=0.05 mass coverage |
|-------|------------------|----------------------|
| 0 | 0.030 | 7.5% |
| 5 | 0.814 | 85.1% |
| 10 | 0.429 | 76.2% |
| 15 | 0.433 | 73.3% |
| 21 | 0.429 | 72.5% |

Row-max ratio (95% mass coverage):

| Layer | Median ratio | Std |
|-------|--------------|-----|
| 5 | 0.0025 | 0.2776 |
| 10 | 0.0064 | 0.0579 |
| 15 | 0.0045 | 0.1367 |
| 21 | 0.0023 | 0.0707 |

## Per-layer active dim sweep

| Layer | min ratio (Spearman ρ ≥ 0.90) |
|-------|-------------------------------|
| 0 | 25.0% |
| 5 | 31.2% |
| 10 | 25.0% |
| 15 | 25.0% |
| 21 | 31.2% |

跨層跨度 6.2 pp

## AWQ Quantization Results

| Config | acc_norm |
|--------|----------|
| FP32 baseline | 59.58% |
| AWQ baseline | 58.34% (drop 1.24 pp) |
| AWQ + RAD + PxV | 57.40% (drop 2.18 pp vs FP32, additive verified) |

MAC counter on AWQ:
- High-prob token: 2.11% (vs FP32 4.1%, attention 變分散)
- 暗示 P_THRES=0.05 在 AWQ 上 sub-optimal

## Known issues

1. **anchor total 3 (2Q+1K) 非 HW-aligned**: 應該重 sweep HW-aligned combos (1+1, 2+2, 1+3, 3+1)
2. **P_THRES=0.05 是 mean-method 估的**: 應該用 per-row binary search 重 measure (跟 Qwen 對齊)
3. **AWQ 重 calibrate P_THRES**: AWQ attention 分布變分散, P_THRES 該 re-measure
