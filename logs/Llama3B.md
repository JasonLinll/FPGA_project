# Llama-3.2-3B-Instruct-AWQ — Calibration 數據附錄 (v6.1)

⚠ Llama-3B-specific 實驗數據, RTL 不寫死, deployment 時 calibration flow 重新生成 register。
⚠ 主 model (D=128, GQA 3:1, ctx 128K); TinyLlama / Qwen 為 cross-model 驗證。
⚠ v6.1 機制: RAD-Cascade (shared-dim active + SIGN + Global Top-N N=64) + KIVI-V + CETT FFN; AWQ INT4 premise; 全棧定點位寬。

---

## 1. Architecture

| 項目 | 值 |
|------|---|
| 模型 | casperhansen/llama-3.2-3b-instruct-awq |
| Layers | 28 |
| Q heads | 24 |
| KV heads | 8 (GQA ratio 3) |
| head_dim | 128 |
| num_pairs | 64 |
| d_ff | 8192 |
| vocab | 128256 |
| context | 128K |

---

## 2. 鎖定配置 (v6.1)

| 參數 | 值 | 依據 |
|------|---|------|
| active dim | **24/64 pair (37.5%)** | 50% 對照上界僅 +0.82% (多買 0.33pp), 省比例優先不採 |
| n_K anchor | **4** | KL sweep {1,2,3,4} 扁平 (max 0.0200-0.0231); e2e n_K=3/4 同值 (+1.16/+1.15%), 槓桿不在 n_K |
| n_Q anchor | 0 | 同 TinyLlama, 冗餘 |
| top_n | f(S): 8/16/32/**64** (S≥1024) | N sweep 證明 N=64 邊際遞減停點 |
| dynamic pair | **GQA shared-dim** (3 Q head/組, Σ\|Q\| 加總選) | per-head union 塌縮 (見 §5), shared 收回 |
| selection 輸入 | **量化前 FP16 幅度** | 純比較零成本, 對症 RAD 選擇敏感 |
| β (SIGN) | 1.0 | Theorem 1 |
| KIVI group | 32 | KIVI default |
| CETT τ* | **0.10** | PPL+1% 標準, e2e 直選 |
| generation V | per-head top-N 截斷, 讀 union | SparQ式, prefill soft-tail |

---

## 3. PG-19 精度 (真跑, S=2048, nc=50, dense<15 凍結清單)

dense baseline = 12.9089 (AWQ INT4, 機制全關)。

### 3.1 attention 全棧 (RAD shared-dim + sel_fp16 + 全位寬 + N=64)
| Config | PPL | vs dense |
|--------|-----|----------|
| AWQ dense | 12.9089 | — |
| attention 全棧 e2e | **13.0568** | **+1.15%** |

### 3.2 CETT-only (τ*=0.10)
| Config | PPL | vs dense |
|--------|-----|----------|
| CETT-only | 13.0114 | **+0.79%** ✓ |

### 3.3 系統合計 (RAD 鎖定 + CETT τ*=0.10)
| Config | PPL | vs dense |
|--------|-----|----------|
| RAD + CETT | **13.1549** | **+1.91%** ≤ 宣告 +2.5% ✓ |

- 加性預期 +1.95% → 實測 +1.91% (**次加性 -0.04pp**, 兩機制正交)。

---

## 4. CETT 鎖定 (FP/per-group-INT8 範數, 均勻 τ)

| 項目 | 值 |
|------|---|
| τ* | 0.10 (PPL+1% 標準) |
| sparsity (真跑) | **44.9%** (> TinyLlama 43.6%; 3B 不難稀疏) |
| FFN MAC 省 | ~15.0% (down 占 FFN 1/3) |
| 範數表落地 | per-group(128) INT8: vs FP **+0.01pp** ✓ |
| 標量 vs 向量 CETT 差 | 10.4% (per-model; τ 僅索引, 選擇以 PPL 為準) |

- **單 scale INT8 範數已驗死**: 3B 吃 1.9pp (8192 範數動態範圍大, 小範數壓爆 → 選擇噪聲)。落地 = per-group(128) INT8。
- 早期 20% 低 sparsity 為單 scale artifact, 修正後 44.9%。

---

## 5. GQA shared selection (union counter, 3:1)

**K dim union (per-head 模式, L5/L20 一淺一深)**:
| 量測 | L5 | L20 | 結論 |
|------|----|----|------|
| K pair union (per-head 選) | 57.2% | 61.0% | per-head 下 K BW 省塌到 ~40% (獨立上界 75.6%, 單 head 37.5%) |

- shared-dim 收回 62.5% dim-level (+~20pp); 跨層一致。
- 3B (3:1) union 沒 TinyLlama (8:1, 82%) 嚴重, 但仍塌縮, shared-dim 必要。

**V token union (per-head top_n=64)**:
| 量測 | 值 | V BW 省 |
|------|---|---------|
| V token union | 118-122 tok (上界 192) | ~94% (/S 口徑) |

- shared top-N 已驗死 (TinyLlama +4.02%, 8 head 候選分歧), 候選保持 per-head, BW 按 union 算。

---

## 6. LongBench L4 終測 (N=200, MAX_LEN=12000, 官方 prompt/maxlen/metric)

dense (AWQ INT4 機制全關) vs RAD+CETT (鎖定配置), 同 generate loop 同截斷。

| 任務 | 類型 | 平均長度 | dense | RAD+CETT | drop | metric |
|------|------|---------|-------|----------|------|--------|
| lcc | code | 2992 | 40.95 | **45.52** | **-4.57** ↑ | edit-sim |
| triviaqa | few-shot | 9475 | 87.72 | **88.54** | **-0.82** ↑ | F1 |
| gov_report | 摘要 | 8562 | 33.48 | 32.96 | +0.52 | ROUGE-L |
| multi_news | 摘要 | 2620 | 25.95 | 25.04 | +0.91 | ROUGE-L |
| multifieldqa_en | 單文檔 QA | 6814 | 48.18 | 46.37 | +1.81 | F1 |
| trec | few-shot 分類 | 6786 | 75.50 | 73.50 | +2.00 | Acc |
| qasper | 單文檔 QA | 4958 | 43.57 | 41.23 | +2.34 | F1 |
| 2wikimqa | 多跳 QA | 6876 | 37.87 | 35.53 | +2.34 | F1 |
| hotpotqa | 多跳 QA | 10756 | 48.14 | 44.98 | +3.16 | F1 |
| **AVG** | | | **49.04** | **48.19** | **+0.85** | |

### 6.1 觀察 (drop 隨任務結構單調)
- **機制近無損**: AVG drop **0.85** (N=200 全量, 比 PPL +1.91% 更貼任務層 evidence)。
- **drop 與「信息分散度」相關** (機制行為證據, 非噪聲):
  - **局部/檢索任務反超或無損**: lcc (code 補全 -4.57) / triviaqa (事實檢索 -0.82) / gov_report·multi_news (摘要 +0.5~0.9)。關鍵信息集中少數 token, RAD top_n 聚焦 = 過濾噪聲 (正則化效應)。
  - **多跳推理損失最大**: hotpotqa (+3.16) / 2wikimqa (+2.34) / qasper (+2.34)。需跨多個分散位置整合, 稀疏 attention 漏中間跳概率高 → 最吃虧。
- **hotpotqa +3.16 為最難場景** (多跳 + 最長 10756 tok), 反映稀疏 attention 對分散信息整合的固有限制; 絕對分 44.98 未崩, 在同類 KV 壓縮方法正常範圍。
- **誠實定位**: RAD 的稀疏行為符合任務語義 (信息集中→受益, 信息分散→小損), 非盲目截斷。

---

## 7. lm_head (已驗死, 維持全讀)

- 兩段式 (SVD sketch + exact rescore): argmax 保全率最佳 68.7% (r=128, C=1024, FP16 sketch), 離 greedy 等價太遠; INT4 sketch 再掉 8pp。
- 128K vocab 低秩殘差太大, SVD-softmax 式不轉移到現代大詞表。
- lm_head 維持全讀 INT4 (203MB/step, 12.3% step BW)。

---

## 8. Exponent window (mantissa-shift lane) — 已實測

激活指數分布 (PG-19, 各 linear pre-input, e_x=floor(log2|x|)):
| linear | max_e | clamp_hi% | clamp_lo% (window [-12,13]) |
|--------|-------|-----------|------------------------------|
| q/k/v_proj | 5 | 0.0000 | 0.34 |
| o_proj | 2 | 0.0000 | 0.88 |
| gate/up_proj | 3 | 0.0000 | 0.13 |
| down_proj | 5 | 0.0000 | **1.26** (最差) |

**window clamp PPL (PG-19 nc=47, 凍結清單; clamp 模擬保守上界=小值全歸0):**
| 配置 | PPL | vs baseline |
|------|-----|-------------|
| 無 clamp | 12.8480 | — |
| **[2⁻¹², 2¹³] (default, 採用)** | 12.8486 | **+0.004%** |
| [2⁻¹⁶, 2⁶] (下移對照) | 12.8474 | -0.005% |

- **clamp_hi = 0 全 linear** (max_e=5, 無上界 outlier; LLM outlier channel 未現於激活指數)。
- **下界 clamp 1.26% (down) 無害**: 極小值 (|x|<2⁻¹²) underflow 歸0, 對 MAC 貢獻微乎其微, 實測 +0.004% (noise floor)。
- **window [2⁻¹²,2¹³] 採用** (下移對照無顯著收益, 不增複雜度)。
- **位寬定案完整**: 全棧位寬 (RoPE16 -0.01% / 定點softmax+U16 +0.02% / V整數MAC 0 / exponent window +0.004%) 全部實測, 無未驗證假設。

## 9. 待補 (pending)
- XSum L4 (定案配置一次, 對齊 TinyLlama 口徑)
- TinyLlama 43.6% sparsity 以 per-group INT8 範數重驗 (口徑對齊, 預期無變)
