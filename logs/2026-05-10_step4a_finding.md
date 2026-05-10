# Step 4A — 維度屏蔽機制驗證與設計修正

**日期**：2026-05-10
**階段**：軟體 Golden Model — 維度顯著性屏蔽（dimension saliency masking）
**核心結論**：原計畫書 §四(二)B 的 per-pair 屏蔽設計實證失敗（Top-1 63%），
              修正為 per-query consistent masking 後驗證成功（Top-1 99.96%）

---

## 1. 動機

原計畫書 §四(二)B 提出維度顯著性屏蔽機制：在 cycle 2 結束後，
利用 BUI 上下界對每個 (Q, K) 對的低貢獻維度動態關閉後續計算。
此機制目標為在 token-level 早停（Step 3, §四(一)A）之上額外提升計算量節省。

## 2. 實驗設定

| 項目 | 內容 |
|------|------|
| Simulator | v5（基於 Step 3 v3 加入維度屏蔽） |
| 採樣 | BERT Layer 3, 20 sample × 12 head = 240 組 |
| 量化 | Per-tensor INT8 |
| 屏蔽時機 | Cycle 2 結束後 |
| Token-level 設定 | K_topk=32, α=0.5, radius_factor=0.05 |
| Metric | Top-1 命中率（最嚴格的排序保真度指標） |

## 3. 初步驗證：機制方向正確

跑 Cell 15 統計驗證「保留 vs 屏蔽維度」的真實貢獻：

| K_significance | 保留維度 mean \|true contribution\| | 屏蔽維度 mean | 比率 |
|----------------|------------------------------------|--------------|------|
| 0 | 2466 | 473 | **5.21×** |
| 2 | 971 | 417 | 2.33× |
| 4 | 824 | 429 | 1.92× |

保留維度的真實貢獻顯著大於屏蔽維度，**屏蔽方向正確**。

## 4. 主要 finding：Per-pair 屏蔽嚴重損害排序

完整 240 組多樣本驗證：

| 設定 | Top-1 命中率 | 對照 v3 (token only) |
|------|------------|---------------------|
| v5, K_sig=2 (per-pair) | **63.24%** | 100.00% |
| v5, K_sig=4 (per-pair) | 62.66% | 100.00% |

Top-1 從 100% 掉至 63%，**排序嚴重損壞**。

## 5. 根本原因分析

實作層面 partial sum 計算經 sanity check 驗證正確：
- 「v3 partial - v5 partial」與「被屏蔽維度應漏掉的真實貢獻」差距 < 5%（30492 vs 30944）
- BUI 推導本身正確（前序 Cell 10 驗證 0 violations）

**問題在「屏蔽粒度」**：
- 每對 (q, k) 的 BUI 不同 → 屏蔽不同維度
- Query q 對 k=3 的 partial sum 算了 47 維、對 k=5 算了 51 維
- 不同維度數的 partial sum **失去可比較性**
- Top-K 排序被任意打亂

## 6. 變體實驗

### 變體 1：延後屏蔽時機

| dim_mask_cycle | Top-1 | dim_kept | Compute save |
|----------------|-------|----------|-------------|
| 2 | 64.0% | 40% | 56% |
| 3 | 72.7% | 26% | 54% |
| 4 | 81.2% | 18% | 46% |
| 5 | 88.5% | 14% | 36% |

延後改善有限，cycle 5 也才 88.5%，**且邊際效益遞減**。

### 變體 2：Per-query consistent masking ⭐

對同一個 query q 的所有 (q, *) 對，取**所有 k 的 BUI union**作為共同屏蔽集合。
這樣 q 的所有 partial sum 都用同一組維度算，保住可比較性。

| K_sig | Per-pair Top-1 | **Per-query Top-1** | dim_kept |
|-------|---------------|--------------------|----------|
| 1 | 61.6% | **99.79%** | 32.8% |
| 2 | 64.0% | **99.96%** | 32.9% |
| 4 | 63.7% | **99.98%** | 32.9% |

Per-query 屏蔽**完全救回 accuracy**，且 K_sig 對結果幾乎無影響（機制穩定）。

## 7. 結論與設計決策

### 主要 finding

「Per-pair BUI dimension masking」在 INT8 bit-serial attention 上**不可行**，
因為不同 (q, k) 對屏蔽不同維度，破壞 partial sum 可比較性。

「Per-query consistent BUI dimension masking」**可行**，保留 BUI 機制的優勢同時
保住 attention 排序。

### 設計決策

- ✅ **保留維度屏蔽機制**，但 granularity 從 per-pair 改為 per-query
- ✅ 計畫書 §四(二)B 改寫為 per-query consistent masking
- ✅ 硬體實作：per-query mask 需要 [seq_q × dim] ≈ 1KB BRAM，
     比 per-pair [seq_q × seq_k × dim] ≈ 1MB 小 128 倍

### 新穎性盤點（誠實版）

| 既有工作 | 機制 |
|---------|------|
| PADE (HPCA 2026) | BUI-GF for **token**-level early stop |
| BitStopper | LATS for **token**-level early stop |
| LeOPARD | Gradient-based **token** pruning + bit-serial |
| SNP | **Offline** structured neuron pruning |
| SpAtten | **Token/head** cascaded pruning |

本計畫的 per-query BUI dimension masking 屬於：
- Runtime（不是 offline）
- Dimension-level（不是 token-level）
- BUI-based（不是 magnitude-based）
- Per-query consistent（不是 per-pair）

此交集據目前 survey 未見現有工作完整實現，**屬於對 PADE 框架的延伸 contribution**。
創新性等級評估：⭐⭐⭐（明確 niche + ablation 證據），不是 paradigm shift。

### 限制與未來驗證

- ⏸️ 目前只在 Layer 3 驗證，需擴展到 Layer 0/3/6/9
- ⏸️ 計算量節省尚未跟 token early stop 完整疊加量化
- ⏸️ End-to-end MNLI accuracy 尚未測試
- ⏸️ 跨模型驗證（GPT-2）尚未進行

## 8. 給未來的我

這個 finding 是經歷以下過程才得到的：

1. 信任計畫書 (per-pair) → 跑出 Top-1 53%
2. AI 助手建議「砍掉機制」→ 我堅持要更多證據
3. 跑 A/B/C 三個變體實驗
4. 發現 per-query 救回 99.96%

**關鍵教訓**：計畫書是 AI 生成的初稿，不是聖旨。實證驗證 + 變體實驗才能找到真實設計。
**研究員的價值在於知道「什麼時候不該相信權威」**——包括計畫書、包括 AI 助手。

## 9. 下一步

- [ ] 重新跑 9 點 sweep（用 v6 = per-query 屏蔽版本）
- [ ] 跨層驗證（Layer 0/3/6/9）
- [ ] Step 4B：二階多項式 Softmax 補償
- [ ] End-to-end MNLI accuracy
