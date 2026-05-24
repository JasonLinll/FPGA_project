# TinyLlama-1.1B-Chat-v1.0 - Calibration 數據附錄 (v4)

⚠ 此檔為 TinyLlama-specific 實驗數據, RTL 不寫死, deployment 時 calibration flow 重新生成 register 配置.

## 1. Architecture

| 項目 | 值 |
|------|---|
| Layers | 22 |
| Q heads | 32 |
| KV heads | 4 (GQA ratio 8) |
| head_dim | 64 |
| num_pairs | 32 |

## 2. Calibrated config (v4 final)

| 參數 | 值 | 說明 |
|------|---|------|
| active_dim ratio | 37.5% (24 of 64) | ρ ≥ 0.90 sweep + SparQ 範圍內 |
| n_Q anchor | 0 | 跨 model 驗證冗餘, 廢除 |
| n_K anchor | 1 | KL max 校準, runtime running-max |
| Dynamic per-query | 11 (= 12 active - 1 K-anchor) | top-K of \|Q\| |
| β | 1.0 | Theorem 1 unbiased optimal |
| block_size | 16 | Chebyshev P(error) < 5% |
| T_margin | 0.75 | 3σ |
| T_overflow | 0.25 | 1σ |
| M_local | 3 | max candidate per block |
| KIVI group size | 32 | KIVI paper default |

## 3. HellaSwag 10042 (acc_norm)

| Config | acc_norm | Drop |
|--------|---------|------|
| Baseline FP32 | 59.58% | - |
| **0Q+1K + runtime_max + SIGN + Block + KIVI (v4)** | **58.60%** | **0.98 pp** |
| 1Q+1K + runtime_max + SIGN + Block + KIVI | 58.61% | 0.97 pp (Q-anchor 冗餘確認) |
| offline K-anchor (1Q+1K) | 58.54% | 1.04 pp |
| EMA α=0.1 (1Q+1K) | 58.41% | 1.17 pp |
| 0Q+0K (SparQ-style, no anchor) | 55.21% | 4.37 pp |

## 4. Component contribution (1Q+1K + runtime_max + KIVI base)

| Config | acc_norm | Drop |
|--------|---------|------|
| C1 full (SIGN + Block) | 58.61% | 0.97 pp |
| C2 no SIGN (Block only) | 57.20% | 2.38 pp |
| C3 no Block (SIGN only) | 54.41% | 5.17 pp |
| C4 minimal (no SIGN, no Block) | 36.46% | 23.12 pp |

Component 貢獻:
- SIGN: +1.41 pp (C1 - C2)
- Block dispatch: +4.20 pp (C1 - C3, dominant)
- Combined synergy: +22.15 pp (super-additive, sum 5.61)

## 5. PG-19 long context (window=2048, stride=1024, 3 books)

| Config | PPL | Δ vs baseline |
|--------|-----|---------------|
| Baseline FP32 | 11.497 | - |
| RAD + KIVI + offline anchor | 12.096 | +5.22% |
| **RAD + KIVI + runtime_max** | **11.961** | **+4.04%** |

Runtime vs Offline: **-1.18% PPL** (runtime 顯著勝 long context, attention sink 自動保護)

## 6. Cross-task (acc_norm, lm-eval-harness style)

| Task | Samples | Baseline | RAD v4 | Drop |
|------|---------|----------|--------|------|
| HellaSwag | 10042 | 59.58% | 58.60% | 0.98 pp |
| ARC-Easy | 570 | 50.18% | 50.18% | 0.00 pp |
| ARC-Challenge | 299 | 31.10% | 29.43% | 1.67 pp |
| PIQA | 1838 | 74.48% | 74.76% | -0.27 pp |

**平均 drop**: 0.60 pp (4 tasks)

## 7. AWQ stack (W4 weight quantization)

| Config | acc_norm | Drop vs FP32 |
|--------|---------|--------------|
| FP32 baseline | 59.58% | - |
| FP32 + RAD v4 | 58.60% | 0.98 pp |
| AWQ baseline | 58.34% | 1.24 pp |
| **AWQ + RAD v4** | **57.72%** | **1.86 pp** |

Sub-additive: AWQ (1.24) + RAD (0.98) = 2.22 predicted, actual 1.86 → AWQ 跟 RAD K-anchor outlier 部分重疊, 機制互補.

## 8. (n_Q, n_K) KL metric validation

KL max 對 (n_Q, n_K) 預測 (4 rep layer × 8 head × 5 long samples):

| Config | KL max | KL mean | end-to-end |
|--------|--------|---------|------------|
| 0Q+0K | 2.1220 | 0.9716 | 55.21% |
| **0Q+1K** | **0.3949** | **0.2677** | **58.60%** |
| 1Q+1K | 0.3949 (identical) | 0.2677 (identical) | 58.61% (tie) |
| 0Q+2K | 0.4723 | 0.2898 | not tested |

**Q-anchor 冗餘確認**: 0Q+1K KL identical to 1Q+1K → end-to-end gap 0.01 pp (noise floor).

## 9. Active dim sweep (Spearman ρ, exclude Layer 0)

| Ratio | min ρ | Pass (ρ ≥ 0.90) |
|-------|-------|-----------------|
| 12% | 0.711 | ✗ |
| 18% | 0.790 | ✗ |
| 25% | 0.847 | ✗ |
| 31% | 0.888 | ✗ |
| **37.5%** | **0.912** | **✓** |
| 50% | 0.947 | ✓ |
| 62% | - | ✓ |

選 37.5% (最小 pass ratio, 在 SparQ 驗證範圍 25-50% 內).

注意: ρ 對 ratio sweep 是「scale 變化」場景, 跟 anchor on/off 不同類, 結合 end-to-end (drop 0.98 pp) 一起驗證才可信.

## 10. Dropped dim distribution (per-layer)

| Layer | Kurtosis | Skewness |
|-------|----------|----------|
| 0 | 25.55 | 3.76 |
| 5 | 0.77 | 0.54 |
| 10 | 0.16 | 0.31 |
| 15 | 0.72 | 0.51 |
| 21 | 0.10 | 0.45 |

Layer 0 特殊 (高 kurtosis) → 異質管線 (跑滿 64 dim).

## 11. Register 設定 (deployment)

```
reg_active_lane_count = 3   // 24 active dim / 8 PE-per-lane
reg_n_K_anchor        = 1
reg_n_Q_anchor        = 0   // default 0, fallback flexibility
reg_block_size        = 16
reg_BETA              = shift 0 (β=1.0)
reg_T_margin          = 0.75 × 256 = 192
reg_T_overflow        = 0.25 × 256 = 64
reg_M_local           = 3
```
