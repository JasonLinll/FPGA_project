# LLaMA 級別硬體加速器 — Algorithmic Constraints

**日期**：2026-05-15
**目的**：明列必須遵守的演算法邊界條件，避免 per-pair 架構的致命缺陷。

---

## Constraint 1：Per-query 維度一致性

- **定義**：同一個 query q 的所有 (q,k) 計算必須使用**同一個 dim mask**。
- **原因**：若 per-pair 各自屏蔽不同維度，partial sum 的量綱不可比較，Softmax 排序會崩潰。
- **實作方向**：採 Q-driven per-query masking，依 Q 的特徵強度（如 MSB 絕對值）生成 q 專屬固定 mask，符合 Data Streaming 無等待特性。

---

## Constraint 2：RoPE 區塊保護（Block-wise Masking）

- **定義**：屏蔽最小粒度為**相鄰 2 維**，以 block 為單位進行保留/屏蔽。
- **原因**：RoPE 以 2 維為一組表示複數相位，若任意拆散將破壞相位連續性。
- **實作方向**：在降低維度數量的同時，維持 RoPE 的相位一致性。

---

## 如何避開傳統 per-pair 架構的致命缺陷

- per-pair 使同一 q 對不同 k 的 partial sum 使用不同維度集合，排序基準失去一致性。
- per-query 固定 mask 將所有 (q,k) 的計算投影到同一降維子空間，可同時維持排序與省電收益。
