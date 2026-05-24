# Qwen2.5-1.5B - Calibration 數據附錄 (v4)

⚠ 此檔為 Qwen-specific 實驗數據, RTL 不寫死, deployment 時 calibration flow 重新生成 register 配置.

## 1. Architecture

| 項目 | 值 |
|------|---|
| Layers | 28 |
| Q heads | 12 |
| KV heads | 2 (GQA ratio 6) |
| head_dim | 128 |
| num_pairs | 64 |

## 2. Calibrated config (v4 final)

| 參數 | 值 | 說明 |
|------|---|------|
| active_dim ratio | 37.5% (48 of 128) | 跟 TinyLlama 同 ratio, SparQ 範圍內 |
| n_Q anchor | 0 | 跨 model 廢除 (TinyLlama + Qwen 驗證) |
| n_K anchor | 3 | KL max 校準, model-dependent (head_dim 128) |
| Dynamic per-query | 21 (= 24 active - 3 K-anchor) | top-K of \|Q\| |
| β | 1.0 | Theorem 1 unbiased optimal |
| block_size | 16 | Chebyshev |
| T_margin | 0.75 | 3σ |
| T_overflow | 0.25 | 1σ |
| M_local | 3 | max candidate per block |
| KIVI group size | 32 | KIVI paper default |

## 3. HellaSwag 10042 (acc_norm)

| Config | acc_norm | Drop |
|--------|---------|------|
| Baseline FP32 | 66.48% | - |
| **0Q+3K + runtime_max + SIGN + Block + KIVI (v4)** | **65.72%** | **0.76 pp** |
| 0Q+1K + runtime_max + SIGN + Block + KIVI | 65.41% | 1.07 pp |

0Q+3K 勝 0Q+1K +0.32 pp, K-anchor 數量 model-dependent.

## 4. (n_Q, n_K) KL metric validation

KL max + KL mean 對 (n_Q, n_K) 預測 (4 rep layer × 8 head × 5 long samples):

| Config | KL max | KL mean | end-to-end |
|--------|--------|---------|------------|
| 0Q+0K | 1.1015 | 0.6941 | not tested |
| 0Q+1K | 1.0406 | 0.6565 | 65.41% |
| **0Q+3K** | **1.0175** | 0.6603 | **65.72%** |
| 1Q+1K | 1.0406 (identical to 0Q+1K) | 0.6565 | not tested |
| 1Q+3K | 1.0175 (identical to 0Q+3K) | 0.6619 | not tested |
| 0Q+4K | 1.0571 | 0.6706 | not tested |

關鍵 finding:
- **Q-anchor 廢除確認**: 0Q+1K KL identical to 1Q+1K, 0Q+3K identical to 1Q+3K
- **KL max 預測對**: 0Q+3K KL max 最低, end-to-end 也最佳
- **KL mean 預測錯**: 0Q+1K KL mean 最低, 但 end-to-end 落後 0.32 pp

→ **KL max 為 primary metric**, KL mean 不可信於同類比較.

## 5. K-anchor 數量 model-dependent

| Model | head_dim | num_pairs | best n_K | n_K / num_pairs |
|-------|----------|-----------|----------|-----------------|
| TinyLlama-1.1B | 64 | 32 | 1 | 3.1% |
| **Qwen2.5-1.5B** | **128** | **64** | **3** | **4.7%** |

經驗法則: n_K ≈ round(0.03-0.05 × num_pairs), 隨 head_dim 增加.

## 6. Register 設定 (deployment)

```
reg_active_lane_count = 8   // 48 active dim / 6 PE-per-lane (差 8 PE 在最後 lane 處理)
                              // 或 6 lane × 8 PE = 48, 完全 fit
reg_n_K_anchor        = 3
reg_n_Q_anchor        = 0
reg_block_size        = 16
reg_BETA              = shift 0 (β=1.0)
reg_T_margin          = 0.75 × 256 = 192
reg_T_overflow        = 0.25 × 256 = 64
reg_M_local           = 3
```

## 7. 待補實驗 (P1)

- PG-19 long context (Qwen 0Q+3K runtime_max vs offline)
- Cross-task (ARC-E/C, PIQA)
- AWQ stack on Qwen
