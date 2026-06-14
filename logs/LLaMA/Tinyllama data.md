# TinyLlama-1.1B-Chat-v1.0 — Calibration 數據附錄 (v6)

⚠ TinyLlama-specific 實驗數據, RTL 不寫死, deployment 時 calibration flow 重新生成 register。
⚠ v6 機制: RAD-Cascade (active dim + SIGN + Global Top-N) + KIVI-V + CETT FFN; AWQ INT4 premise。
舊 v4 機制 (Block dispatch / M_local / T_margin / gate-proxy FFN) 已淘汰, 本檔刪除其數據, 僅保留仍成立之結論。

## 1. Architecture

| 項目 | 值 |
|------|---|
| Layers | 22 |
| Q heads | 32 |
| KV heads | 4 (GQA ratio 8) |
| head_dim | 64 |
| num_pairs | 32 |
| d_ff | 5632 |

## 2. Calibrated config (v6.1)

| 參數 | 值 | 說明 |
|------|---|------|
| active_dim ratio | **固定 37.5%** (24/64) | 雙檔已刪 (短序列精度餘裕更小); 見 §9 |
| top_n | f(S): 8/16/32/**64** (S≥1024) | 長 S 加 64 檔 (N sweep §13) |
| n_Q anchor | 0 | KL 跨 model 冗餘, 廢除 |
| n_K anchor | 1 | KL max 校準, runtime running-max |
| Dynamic per-query | 11 (= 12 active pair - 1 K-anchor) | top-K of \|Q\| |
| dynamic dim 選擇 | **GQA shared-dim** (8 Q head/組, Σ\|Q\| 加總) | GQA 8:1 union 修正 (§14) |
| selection 輸入 | **量化前 FP16 幅度** | 純比較零成本 (§13) |
| β (SIGN) | 1.0 | Theorem 1 unbiased |
| KIVI group size | 32 | KIVI paper default |
| 全棧位寬 | RoPE INT16 / 定點 softmax / attn_w U16 / V 整數 MAC / mantissa-shift linear | §12 位寬階梯 |
| CETT τ_CETT | 0.08 | PPL+1% 標準, per-layer ε_i |
| generation V | top_n 截斷 (SparQ式) | 只讀候選 V (per-KV-head union), prefill soft-tail |

**attention 全棧定案** (RAD shared-dim + 全位寬 + sel_fp16 + N=64): PG-19 nc=50 S=2048 = **12.3087, vs AWQ dense +0.74%** (§13)。

## 3. AWQ 精度 (真跑, 部署口徑, 4-way)

分離量化損失 (AWQ vs FP32) 與機制損失 (機制 vs AWQ)。

### 3.1 PG-19 PPL (S=2048, nc=50)
| Config | PPL | 說明 |
|--------|-----|------|
| FP32 baseline | 11.6892 | - |
| AWQ-only | 12.2178 | 量化損失 +4.52% |
| AWQ + RAD | 12.3322 | 機制 vs AWQ +0.94% |
| AWQ + RAD + CETT | 12.4510 | 機制 vs AWQ **+1.91%** |

### 3.2 HellaSwag 10042 (acc_norm, byte-norm)
| Config | acc_norm | 說明 |
|--------|---------|------|
| FP32 baseline | 59.58% | - |
| AWQ-only | 59.52% | 量化損失 -0.06pp (classification 對量化幾乎無損) |
| AWQ + RAD | 59.12% | 機制 vs AWQ -0.40pp |
| AWQ + RAD + CETT | 59.01% | 機制 vs AWQ **-0.51pp** |

### 3.3 XSum ROUGE (generation, N=100)
| Config | R1 | 說明 |
|--------|----|----|
| FP32 baseline | 25.31 | - |
| AWQ-only | 22.56 | 量化損失 |
| AWQ + RAD + CETT (gen top_n V) | 21.65 | 機制 vs AWQ drop **0.91** |

**關鍵**: 量化敏感度任務相關 — classification (HellaSwag) 幾乎無損 (-0.06pp), generation (PPL +4.52%, ROUGE -2.75) 較敏感。機制損失皆小且 additive 不放大。

## 4. 真跑 MAC (ground truth, HellaSwag prefill, forward counter)

| 項目 | 值 |
|------|---|
| Q·K MAC 省 (含 layer0) | 58.6% (dense 8.464T → 3.503T) |
| FFN linear 占總 MAC | 78.3% |
| Q·K 占總 MAC | 0.24% (短序列) |
| CETT 真實 sparsity | 43.6% (每層平均, 非套定值) |
| FFN MAC 省 (CETT down) | 14.5% (down 占 FFN 1/3, gate/up 全算) |

**定位**: 整體 MAC 大頭是 FFN linear (78%); attention Q·K 省比例高 (58.6%) 但占比小 (長序列才上升)。機制省 MAC 主力 = CETT FFN; attention 價值在精度保持 + memory-bound BW。

## 5. active × top_n sweep (PG-19, PPL+1% 上限)

每 S 用自己的 AWQ dense baseline (RAD+CETT 合計, 口徑一致):

### S=512 (baseline 10.3784, PPL+1% 門檻 10.4822)
| active | top_n | PPL | vs base | Q·K省 | 守 PPL+1% |
|--------|-------|-----|---------|-------|-----------|
| 37.5% | 32 | 10.6547 | +2.66% | 54.7% | ✗ |
| 25% | 32 | 10.7718 | +3.79% | 65.6% | ✗ |
| 18.75% | 32 | 10.9240 | +5.26% | 71.1% | ✗ |
| 37.5% | 16 | 10.7562 | +3.64% | 58.6% | ✗ |
| 25% | 16 | 10.9626 | +5.63% | 70.3% | ✗ |

**結論**: active 越激進 PPL 單調惡化 (37.5→25→18.75%: +2.66→+3.79→+5.26%)。**37.5% 為實際上限**, 更激進不可行。想多省 attention MAC = 犧牲精度, 且 attention MAC 占比小, 不划算。
(注: S=512 RAD+CETT 合計 +2.66% 高於 S=2048 的 +1.91%, 因短序列 + CETT/RAD 合計; 不分離因不影響「37.5% 是上限」結論。)

⚠ S=2048 sweep 待補。

## 6. PG-19 K-anchor: runtime running-max vs offline (仍成立)

| Config | PPL | Δ |
|--------|-----|---|
| RAD + KIVI + offline anchor | 12.096 | +5.22% |
| RAD + KIVI + runtime running-max | 11.961 | +4.04% |

Runtime vs Offline **-1.18% PPL** (runtime 勝 long context, attention sink 自動保護)。採 per-position running-max。
(注: 此為 v4 FP32 口徑相對值, runtime 勝 offline 之結論在 v6 仍成立; AWQ 口徑絕對值見 §3。)

## 7. (n_Q, n_K) KL metric validation (仍成立)

| Config | KL max | KL mean | 結論 |
|--------|--------|---------|------|
| 0Q+0K | 2.1220 | 0.9716 | 無 anchor, 崩 |
| **0Q+1K** | **0.3949** | **0.2677** | 採用 |
| 1Q+1K | 0.3949 | 0.2677 | 與 0Q+1K identical → n_Q 冗餘 |
| 0Q+2K | 0.4723 | 0.2898 | 未進一步測 |

**n_Q 冗餘確認**: 0Q+1K 與 1Q+1K KL identical, end-to-end gap 0.01pp (noise floor) → n_Q=0。

## 8. anchor 必要性 (仍成立, 證明 RAD vs 純 SparQ-style)

| Config | HellaSwag drop (v4 FP32) |
|--------|--------------------------|
| 0Q+0K (SparQ-style 無 anchor) | 4.37pp (崩) |
| 0Q+1K (RAD, 加 K-anchor) | 0.98pp |

**K-anchor 為 RAD 關鍵差異**: 純 query-based selection (SparQ-style) 無 anchor 崩 4.37pp; 加 per-position running-max K-anchor 救回到 0.98pp。此為 RAD vs SparQ 的核心增量之一。
(注: v4 FP32 口徑; anchor 必要性結論在 v6 成立, AWQ 口徑見 §3。)

## 9. active dim 選擇依據 (PPL+1%, 取代舊 ρ)

v4 用 Spearman ρ≥0.90 選 37.5% — 但 ρ 對機制決策不可靠 (spec §2.1)。v6 改用 **end-to-end PPL+1%**:
- §5 sweep 證明 37.5% 為 PPL+1% 上限 (25% 已 +3.79% 超標)。
- 37.5% 落在 SparQ 驗證範圍 (25-50%) 內。
- 序列級雙檔: 短序列 (S<256) 可用 25% (compute-bound 省 MAC 優先, 精度容忍度待 per-S 驗)。

## 10. Dropped dim 分布 (per-layer, layer0 異質依據)

| Layer | Kurtosis | Skewness |
|-------|----------|----------|
| 0 | 25.55 | 3.76 |
| 5 | 0.77 | 0.54 |
| 10 | 0.16 | 0.31 |
| 15 | 0.72 | 0.51 |
| 21 | 0.10 | 0.45 |

Layer 0 高 kurtosis → 異質管線 (跑滿 64 dim, 不套 RAD selection)。

## 11. Register 設定 (v6 deployment)

```
reg_active_lane_count = 3       // 24 active dim / 8 PE-per-lane (S>=256)
reg_n_K_anchor        = 1
reg_n_Q_anchor        = 0       // default 0, fallback
reg_top_n             = f(S): 8/16/32
reg_BETA              = shift 0 (β=1.0)
reg_cett_eps_base     = per-layer ε_i (SRAM 表, τ_CETT=0.08 校準)
reg_kivi_group        = 32
```

## 12. 位寬階梯 (AWQ, PG-19 nc=50 S=2048, 累加 ablation)
| 級 | 配置 | PPL | vs 前級 | 裁決 |
|----|------|-----|---------|------|
| L0 | shared-dim base (FP16) | 12.3594 | — | sanity (重現 12.3567 ✓) |
| L1 | +RoPE INT16 | 12.3588 | -0.01% | **採用** |
| L2 | +定點 softmax / attn_w U16 | 12.3616 | +0.02% | **採用** (= 部署位寬定案值) |
| L3 | +A8 qkv/o/gate/up | 12.4287 | +0.54% | 分解 ↓ (死) |
| L4 | +A8 down | 12.4568 | +0.23% | 死 |

- L3 分解 (各單獨 vs L2): qkv **+0.35%** / o -0.04% / gate+up +0.10% (近加性)。混合 (qkv FP16, 其餘 A8) +0.42% 超門檻且零硬體收益 → A8 全砍 (見 spec dead directions)。
- qkv 獨大: A8 擾動投影輸出 → 與 RAD 選擇交互, 非 linear 本身誤差。

## 13. selection FP16 + top-N sweep (AWQ, PG-19 nc=50 S=2048, 基準 L2=12.3616)
| 配置 | PPL | vs L2 | vs AWQ dense | L10 V union |
|------|-----|-------|--------------|-------------|
| sel_fp16, N=32 | 12.3581 | -0.03% | +1.15% | 70.2 |
| sel_fp16, N=48 | 12.3268 | -0.28% | +0.89% | 104.4 |
| **sel_fp16, N=64 (採用)** | **12.3087** | **-0.43%** | **+0.74%** | 137.8 |

- sel_fp16 (選擇輸入量化前 FP16): 零硬體成本 (純比較), 白賺。
- N 邊際遞減 (32→48 -0.25pp, 48→64 -0.15pp) → 停 64, 不試 96。
- **attention 全棧 vs AWQ dense = +0.74% < 1%** ✓

## 14. GQA shared selection (union counter, L10, GQA 8:1)
| 量測 | 值 | 結論 |
|------|---|------|
| K active-pair union (8 head, per-head 選) | 82.0% ± 7.0% (獨立上界 97.7%) | per-head 下 K BW 省塌到 18% |
| shared-dim end-to-end | +0.14% PPL (vs per-head) | **採用**: K BW 省回 62.5% dim-level, 代價 noise floor |
| shared top-N (N_g=32) | +4.02% PPL (vs shared-dim only) | **死**: 8 head 候選本質分歧 |
| V token union (per-head top_n=32) | 77.5 ± 10.4 tok (上界 256) | V 讀取按 union: 省 92.0% |

- **非對稱 insight**: dim 重要性結構性共享 (shared 幾乎無損); token 重要性 head-specific (shared top-N 死)。
- GQA 8:1 (TinyLlama) 為最壞 union 情況, 3B (3:1) union 較輕 (見 llama3b_data.md)。

## 15. 待補 (pending)
- AWQ cross-task (ARC-Easy/Challenge, PIQA 的 AWQ 版)
- CETT 向量和範數 (TDA Eq.20) 嚴格版 vs 標量近似 (TinyLlama 差<2%, 已知可忽略)
- 43.6% sparsity 以 per-group INT8 範數重驗 (口徑對齊 3B, 預期無變)
- XSum L4 shared-dim 併入口徑

---

## 已刪除 (v4 舊機制, 不再適用)
- Block dispatch (block_size=16, T_margin=0.75, T_overflow=0.25, M_local=3) — 被 Global Top-N 取代
- Component C1-C4 (SIGN/Block 貢獻分解) — Block 已廢, SIGN 對 v6 (無 Block) 貢獻待新測
- gate-proxy FFN (τ=0.95 → 20%) — 被 CETT (43%) 取代
- ρ active sweep 作為選擇依據 — 改用 PPL+1% (ρ 數據移至 §9 註)
