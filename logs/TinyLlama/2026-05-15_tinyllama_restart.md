# TinyLlama Restart — Baseline / Q-Driven Masking / RoPE Block Protection

**日期**：2026-05-15
**目的**：重建 TinyLlama 驗證路線，承接已驗證的 BERT-base finding，並明確列出 TinyLlama 需重新驗證的項目與限制。

---

## 計畫基本資訊

- **Target**: KV260 (117K LUT, 1248 DSP, 4GB DDR4, 5.1Mb BRAM, $199)
- **Benchmark**: TinyLlama-1.1B on HellaSwag(主) + BERT-Base on MNLI(副)
- **Toolchain**: Vivado, Verilator
- **對標論文**: PADE (HPCA 2026, arXiv:2512.14322), BitStopper, LeOPARD, SpAtten, ViTCoD, SNP, LLM.int8(), SmoothQuant

---

## 已驗證的 5 個 ⭐⭐⭐ Finding（BERT-base）

### Finding 1: Per-query consistent dimension masking
- 原 per-pair 屏蔽（每個 (q,k) 對各自選 dim）實證失敗
  - 不同 (q,k) 對屏蔽不同 dim → partial sum 量綱不同 → 排序壞
  - Top-1 從 100% 掉到 63%
- 修正：per-query union（對同一 q 的所有 valid k 取屏蔽集合 union）
- 結果：Top-1 99.96%，計算量節省 49.4–55.6%，dim_kept 40%
- 跨 4 層 × 9 設定（K_topk, alpha, radius_factor）全部驗證 PASS

### Finding 2: End-to-end BERT MNLI 0.10pp accuracy loss
- BERT-Base 12 層全套 v6（token + per-query dim, INT8）
- 1000 sample：baseline 85.00% vs v6 85.10%（drop -0.10pp）
- Root cause：HuggingFace eager_attention_forward 內部做 .transpose(1,2) 後才 reshape
  - 手寫 forward 漏掉，導致 logits 壓平

### Finding 3: INT3 attention scores cliff
| Bits | Levels | Accuracy | Drop |
|------|--------|----------|------|
| 8    | 256    | 84.90%   | 0.10pp |
| 4    | 16     | 85.00%   | 0.00pp |
| 3    | 8      | 85.00%   | 0.00pp |
| 2    | 4      | 53.50%   | 31.50pp ← cliff |

- BERT-Base attention scores 只需 8 levels 就保留全部資訊
- 原計畫書 Softmax 補償設計因此失去舞台

### Finding 4: K_topk=8 跨層最優
- Grid search (K_topk × alpha) 跨 4 層：best 都是 K=8, alpha=0.3
- v6 conservative BUI 機制天然 layer-aware，外加 per-layer parameter 冗餘
- 從 K_topk=32 → 8，compute saving 53% → 56–58%

### Finding 5: v6 跨架構 generalize
- BERT-Base (encoder, MNLI 1000)：drop -0.10pp
- TinyLlama-1.1B (decoder, HellaSwag 200)：drop -1.00pp
- 不同架構 + 不同 task + bidirectional vs causal attention 都 work

---

## 補充 finding：probs 量化敏感度高於 attention scores

- probs INT4：drop 1.80pp ❌
- probs INT5：drop 0.10pp ✓
- 但 INT5 無 DSP packing 優勢，對應 asymmetric DSP 機制死

---

## 已嘗試但失敗的 7 個方向（請避開）

1. **Layer-adaptive parameter**：4 層 grid search 完全一致 → BUI 自帶 layer-aware
2. **Poison + Antidote**：2-bit BUI 寬度 ≈ 516,000，真實 score 範圍 140 → 全部 overlap
3. **Delta encoding**：BERT ΔQ/Q ratio 0.78–1.07 → LayerNorm 打散，|ΔQ|≈|Q|
4. **Asymmetric DSP allocation**：probs INT4 掉 1.80pp，INT5 無 packing 優勢
5. **Q-driven masking（僅用 |Q|）**：BERT L3 Q≥8 (Top-1 96%, dim_kept 78%) / Q≥16 (88%, 57%) < per-query union (99.96%, 40%)
6. **Per-pair normalization**：多種 correction factor (0.0/0.192/0.5/1.0) Top-1 掉到 25–62%
7. **Outlier backbone**：TinyLlama 跨層 Top-1 match 38–91% 差異過大，不可做全模型機制

---

## TinyLlama 現有 baseline（200 samples）

- **Baseline**：43.50%（45.0s）
- **v6**：44.50%（111.1 min）
- **Drop**：-1.00 pp
- 註：TinyLlama 官方 HellaSwag 約 59.2%

---

## TinyLlama 下一步（依目前問題敘述）

1. **驗證 Baseline 失敗主因**
   - 讀取 per-pair masking 記錄
   - 對比同一個 q 對不同 k 的 dim_kept 差異
   - 以數據證明「partial sum 量綱不匹配」導致 Softmax 排序崩潰
2. **改寫屏蔽核心邏輯（Q-Driven, Per-query）**
   - 捨棄聚合 k 的作法
   - 只依 Q 的特徵強度（如 MSB 絕對值）生成 q 專屬固定 mask
   - 確保同一 q 的所有 (q,k) 在同一降維子空間計算，符合 Data Streaming 無等待
3. **引入 RoPE 保護機制**
   - 實作 Block-wise masking（最小粒度相鄰 2 維）
   - 避免 RoPE 相位資訊被撕裂
4. **重新驗證與參數掃描**
   - 跑 240 組樣本驗證
   - Sweep K_sig 或 Q 閾值參數
   - 觀察 Top-1 命中率是否恢復、dim_kept 是否落在 40–50% 省電區間
5. **更新架構設計文件**
   - 已新增 constraint 文件：`logs/TinyLlama/architecture_constraints.md`

---

**備註**：BERT-base 成立的機制不保證 TinyLlama 也成立，需要重新驗證。
