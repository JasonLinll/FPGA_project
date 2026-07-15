# RTL 檔案結構 Spec (整併版, 7 檔)

整併原則: **零改碼** — 各來源檔逐字保留為段落 (含各自 header/timescale)，僅加檔頭與
`// ===== 來源: xxx.v =====` 分隔。行尾正規化為 LF。模組名全域唯一，七檔可聯編。

> **本版異動 (decode attention 整合輪)**
> - C2 (softmax runtime scale) / C3 (stage1 因果縫接) 定案並驗證。
> - 新增 4 件並驗證: `ktail_shadow` / `sk_ram` (→ KV_ddr.v)、`k_feeder` / `krunmax_bank` (→ KV_frontend.v)。
> - **補登 3 個既存但漏列的模組**: `q_frontend` (→ KV_frontend.v)、`rsqrt_seed_nr` / `rmsnorm_unit` (→ 新增 rmsnorm.v 第 7 檔)。
> - 缺件表移除「s_k 片上 RAM」「SCALE_MUL CSR」(已補齊)。
> - `softmax_unit` 由 elaboration 常數 scale 改為 runtime mant+exp bank (見 §5)。

---

## 1. KV_frontend.v — 量化 / RoPE 前端子系統

| 模組 | 來源 | 角色 |
|---|---|---|
| `rope_lut_gen` | rope.v | RoPE cos/sin LUT 生成 (無自初始化: theta_tab/cos_lut/sin_lut 外部載入) |
| `rope_unit` | rope.v | 逐對旋轉 (Q.16)，K 路與 Q 路共用 |
| `recip_seed_nr` | K_frontend.v | 倒數: seed LUT (256×16b) + 1 次 Newton-Raphson; LZD e/m 分解一物三用 (recip 尾數索引 / e_sk / m_sk)。**K 與 V 量化共用** |
| `k_quant_unit_v2` | K_frontend.v | K per-token INT8 量化 (scale=absmax/128)，ping-pong 雙 buffer 配 RoPE 收流；輸出 k_int8 / sign plane / s_k (mant+exp) |
| `k_pipeline_top` | K_frontend.v | rope→quant 串接頂層 + runmax 併流 (供 RAD anchor)；**pair_mag 匯出口** (FP16 保序值, 單一轉換源, 供 krunmax_bank；additive, 既有行為零變) |
| `v_quant_unit` | V_quant_unit.v | V per-token KIVI INT4 (G=32): min/max 掃 → 15/range 倒數路 → vq/m_s/e_s/m_q412；range==0 合法路；s 分解走全精度 range×4369 再 LZD |
| **`k_feeder`** ✨ | k_feeder.v | **K 前端 vec→RoPE 拆流** (C1 缺件補齊)。整向量 pulse 鎖存 → 1 pair/cycle 拆流 → rope_lut_gen+rope_unit → k_pipeline_top 直插。協議與 q_frontend 前段同構 (REQ→WT 非重疊)。註: Rtl_spec 舊稱 k_pipeline_top「含 rope」實則其輸入為 RoPE 後 Q.16 逐對流, k_feeder 補上游 |
| **`krunmax_bank`** ✨ | krunmax_bank.v | **per-KV-head K running-max 儲存** (多 head decode)。吃 k_pipeline_top pair_mag 匯出口; RMW 2 段流水 (含 write-forward 防同址冒險) + per-head load-to-flat + clear 全掃。HKV×P×15b ≈ 1 BRAM。取代 k_pipeline_top 單 head runmax_reg 的多 head 混值 |
| **`q_frontend`** ✨補登 | q_frontend.v | **Q 前端** (per GQA 組)。linear_engine out_flat Q 向量 → ① 拆對 rope → ② k_quant_unit_v2 **零改動重用** (INT8 absmax/128, pair_abs 懸空) → ③ q_int8 → Q bank 寫脈衝 (attn_chain_* q_wr 口徑) → ④ q_pairmag_unit 餵 beat (組首/尾) → pairmag_flat (RAD dyn 選擇) → ⑤ s_q → SM 值 (mant+exp 正規化, SM_C=33434=log2e/√128 摺算; SFS=12 記帳差由 attn_chain SM bank 寫入時 exp-12 摺)。Q 側對等於 K 路 k_feeder (rope_unit 共用)。例化 RAD_attn 的 q_pairmag_unit → 聯編需 RAD_attn.v |

## 2. KV_cache.v — 片上 KV 儲存子系統 (demo profile)

| 模組 | 來源 | 角色 |
|---|---|---|
| `k_cache` | k_cashe.v | K 三平面: pair-major dim 平面 (port A stream) + pair-major sign 平面 (port B) + row-major (port C gather, s_k 17b 隨行)。三平面物理分離零仲裁；BW counters 入 assert |
| `v_cache` | V_cache.v | V row-major 單平面 (644b/row = Vq 512 + ms/es/mq 132)，讀出即 v_engine 餵入格式 |

## 3. KV_ddr.v — DDR 存取 + 多 head 排程子系統 (部署 profile)

| 模組 | 來源 | 角色 |
|---|---|---|
| `addr_gen` | gather_engine.v | 六模式位址生成 (K_ROW/K_DIM/K_SIGN/V_Q/V_SC/W_DOWN)，runtime stride/base |
| `gather_fsm` | gather_engine.v | 滑動窗口 (DEPTH=16) tag 亂序重排 → 順序輸出 |
| `kv_ddr_writer` | Kv_ddr_writer.v | 寫側 packer (多 head): K_row/V 直寫 burst；K_dim/K_sign 走 per-head write-combine staging (T_BLK=64→64B/dim=1 burst)；staging 尾巴讀口 (stage1 因果縫接)；flush_tail 逐 head 掃 |
| `kv_gather_top` | Kv_gather_top.v | 單 index stream 批次頂層: 一份 list + 一套 FSM/DMA，K/V 分 phase (`go_k`/`go_v`)；V_SC 5B/group 拆欄 |
| `kv_rd_sched` | Kv_rd_sched.v | 讀側 per-head 輪詢: 灌 list → go_k → consume_k_done → go_v → consume_v_done → 下一 head；空 head 跳過 |
| `kv_req_bridge` (v3) | Kv_req_bridge.v | attn_score_core 串流 req ↔ gather 批次橋: S1 走 kdim_prefetch 本地合成 (pf_enable=1) / S2F 走 FIFO+chunk 真 row；per-entry KV head (cfg_kv_head, req_head 是 GQA Q-slot 不可定址)；sk mux 跟回應源；pf_enable=0 = v1 |
| `kdim_prefetch` | kdim_prefetch.v | 定案 (a) stage1 Active-K 預填: pair_mask → K_DIM/K_SIGN 條帶 session (idx 即位址) → dim-slot banks → 1-cycle 合成 row (active←真值, dropped←sign±1)；3Q 共用一次 fill。**A_MAXP 上限由頂層參數化 (attn_chain_ddr_top 設 32, 涵蓋鎖定 24 + L0=32)** |
| `gqa_union` | Gqa_union.v | GQA 3:1 查詢側歸屬: 3 Q-head Top-N → 去重 union (bitmap+epoch, 回卷掃清) + mask 查詢 + qpos 映射；drain 直驅 kv_rd_sched |
| **`ktail_shadow`** ✨ | ktail_shadow.v | **staging 尾巴 K row 影子** (C3 stage1 因果縫接)。decode 當前 token 的 K 尚在 writer staging 未落 DDR → 攔 bridge→prefetch 的 srv 流, 窗口內 (t∈[stg_base, stg_base+stg_cnt)) 由片上鏡射 1-cycle 合成 (prefetch 同式), 窗口外透傳。0 額外 cycle。寫側同拍鏡射整 row (int8+sign), HKV×T_BLK ≈ 2 URAM。索引 = n_tok_k_flat[head] mod T_BLK (writer 單一真值源) |
| **`sk_ram`** ✨ | sk_ram.v | **s_k side-plane 片上 BRAM** (DDR profile 專用)。缺件補齊: 片上 profile s_k 活在 k_cache 內, DDR profile 需獨立常駐片上模組。寫口 = K 路 tok_valid 拍 ({mant[15:5], exp} 縮位約定同 k_cache); 讀口 = bridge sk_req/sk_idx (1-cycle sync)。HKV×S_MAX×17b ≈ 8 BRAM36 |

## 4. RAD_attn.v — RAD 注意力核心 (原樣未動)

| 模組 | 角色 |
|---|---|
| `topk_engine` | 串流 Top-N 選擇 (cand_idx) |
| `qk_pu` | Q·K 對級 PU: acc_act (active) + acc_sign (dropped, Σ\|q\|·sign(k)) 雙累加 |
| `score_fixup` | F1 定點: act·m_sk·2^(e_sk+F_S-10) + (acc_sign>>>β_sh)<<<F_S → Q.F_S=6。**s_k 側摺算 (s_q 側改由 softmax_unit runtime scale 摺, 見 §5)** |
| `krunmax_unit` | K running-max (anchor 選擇源) |
| `q_pairmag_unit` | Q pair 幅值 (dynamic 選擇源) |
| `attn_score_core` | stage1 全掃 + Top-N + stage2 全精度重算；**內建 GQA 組迭代 (g_q) 與候選選擇**；req/data 串流介面 (按序)；mode=1 硬截斷 / mode=0 soft-tail |

## 5. attn_chain.v — 注意力消費鏈子系統

| 模組 | 來源 | 角色 |
|---|---|---|
| `softmax_unit` | softmax.v | 定點 base-2 softmax。**runtime scale (mant+exp) 語義 (C2 定案)**: 刪常數 SCALE_MUL 乘法路, 改 `scale_mant`(Q1.10 hidden-1)+`scale_exp`(signed 8b) 於 start 拍採樣, EXP 路雙向 barrel shift。摺 s_q (per-token per-head); s_k 由 score_fixup 摺。FP16 aw (mant=0 精確標記 aw=0) |
| `attn_chain_top` | Attn_chain_top.v | decode 消費鏈頂層 (**片上 profile**): score core ↔ k_cache portC 橋 (+Q bank) → softmax → **aw≠0 過濾** V feeder → v_engine；consume_k/v_done 脈衝 (kv_rd_sched 口徑)。**含 SM bank** (per Q-slot scale 寫口, hh mux; reset 預設 = SM_SCALE_MUL 參數換算 = legacy bit-exact) |
| `attn_chain_ddr_top` | attn_chain_ddr.v | decode 消費鏈頂層 (**DDR profile**): K 路 = kdim_prefetch **+ ktail_shadow** + bridge + gather#K (3 DDR master, AXI interconnect 假設)；V 路 = awb list → gather#V go_v → v_engine；attn_out 與片上版 **bit-exact** (tb_attn_chain_ddr 雙 profile 對拍)。**含 SM bank + shadow 寫側/窗口口 (kv_ddr_writer 直插; 無 writer 的 legacy tie: stg_base=s_len, stg_cnt=0)** |

## 6. gemv_engine.v — GEMV / attn·V 引擎 (原樣未動)

| 模組 | 角色 |
|---|---|
| `gemv_lane` | INT 乘加 lane |
| `psum_premul_acc` | 部分和預乘累加 |
| `linear_engine` | 投影 GEMV (輸出 Q.16, 24b, 整向量 pulse — v_quant 輸入源) |
| `v_engine` | attn·V: Σ(aw·s)·Vq + Σaw·m 係數摺疊，Q3.20 輸出 |
| `fp_normalize` | Q3.20 → FP16 |

## 7. rmsnorm.v — RMSNorm 前置級 ✨補登

| 模組 | 來源 | 角色 |
|---|---|---|
| `rsqrt_seed_nr` | rmsnorm.v | 倒數平方根 1/√S: LZD + seed LUT (64 entry) + 1 次 Newton-Raphson。start 後 4 拍 done, 輸出 m_r (Q2.16, ∈(0.5,2]) + e_r。奇 k 摺一位; r=√N·y1·2^(-k'/2) (√N 摺常數 SQN, N 免除法)。**Rule 2: 無除法器** |
| `rmsnorm_unit` | rmsnorm.v | 兩 pass 串流: Σx² → rsqrt → 逐元素 x·w·r 正規化。輸出口徑 = linear_engine L1 活化輸入 (per-element mx Q1.10 hidden-1 / ex s6 / x_sign), RMSNorm 後直餵 QKV/FFN GEMV, 免 INT8 量化 pass (r 為公因子, 只進指數/尾數正規化不進碼值)。精度 FP11 (2^-11) ≫ NR 誤差 |

> **歸屬**: GEMV 上游前置級 — 不屬 KV 前端 (非量化/RoPE) 亦不屬 gemv_engine (非純算子)。
> linear_engine 的活化輸入源, 自成子系統。

---

## 資料流 (decode step)

```
[rmsnorm] RMSNorm → [gemv_engine] linear_engine (QKV proj) ─┬→ [KV_frontend] q_frontend (Q 路)
                                                             ├→ [KV_frontend] k_feeder (K 路)
                                                             └→ [KV_frontend] v_quant (V 路)

[KV_frontend]  k_feeder(vec→rope) → k_pipeline(rope→k_quant) ─┬→ [KV_cache] k_cache      (片上 profile)
                                     │                        └→ [KV_ddr]   kv_ddr_writer (DDR profile)
                                     ├→ pair_mag → [KV_frontend] krunmax_bank (per-head runmax)
                                     └→ s_k ─────→ [KV_ddr] sk_ram (DDR profile 片上 side-plane)
[KV_frontend]  q_frontend → Q bank + pairmag (RAD dyn) + SM 值 (→ attn_chain SM bank)
[gemv_engine]  linear_engine → [KV_frontend] v_quant ─┬→ v_cache / kv_ddr_writer

[RAD_attn] attn_score_core ──req──> k_cache portC (片上)
                            └─req──> [KV_ddr] kv_req_bridge → { kdim_prefetch → ktail_shadow (窗口縫接) } / kv_gather_top (DDR)
           f 流 → [attn_chain] softmax (runtime SM bank scale) → aw 過濾 → v_cache / gather V phase
                                        → [gemv_engine] v_engine → attn_out
多 head: [KV_ddr] gqa_union (3Q→union) → kv_rd_sched (per-KV-head 輪詢)
         runmax = krunmax_bank load[g]; s_q scale = SM bank[g_q]
```

## TB → 檔案對照 (整併後編譯清單)

| TB | 編譯來源 | 狀態 |
|---|---|---|
| tb_k_cache / tb_v_cache | KV_cache.v | |
| tb_integ_kcache | KV_cache.v KV_frontend.v | |
| tb_v_quant | KV_frontend.v | |
| tb_integ_vchain | KV_frontend.v KV_cache.v gemv_engine.v | |
| tb_kv_gather / tb_kv_ddr_writer / tb_kv_writer_mh / tb_kv_rd_sched / tb_gqa_union / tb_kv_req_bridge | KV_ddr.v | |
| **tb_sk_ram** ✨ | KV_ddr.v | ALL PASS |
| **tb_ktail_shadow** ✨ | KV_ddr.v | ALL PASS (8) — 真 writer+prefetch+DDR 整合; Phase A staging vs Phase B 純 DDR 共用 golden |
| **tb_k_feeder** ✨ | KV_frontend.v | ALL PASS (7 tok) — RoPE bit-exact + 獨立三角真值錨 + 假通過防護 (LUT 自算, 非零守衛) |
| **tb_krunmax_bank** ✨ | KV_frontend.v | ALL PASS — 多 head 交錯 RMW + load 快照 + clear + 同址 forward |
| **tb_softmax_rt** ✨ | attn_chain.v RAD_attn.v gemv_engine.v KV_ddr.v | ALL PASS (11) — runtime scale 整數 golden + legacy 等價錨 + 移位邊界 |
| **tb_top_wiring** ✨ | attn_chain.v RAD_attn.v gemv_engine.v KV_ddr.v | ALL PASS (4) — SM bank 預設/摺算/mux + shadow tie |
| tb_attn_chain | attn_chain.v RAD_attn.v gemv_engine.v KV_cache.v KV_frontend.v | 需回歸 (SM bank; 不寫 bank = legacy bit-exact) |
| tb_stage1_equiv | KV_ddr.v RAD_attn.v | |
| tb_attn_chain_ddr | attn_chain.v KV_ddr.v RAD_attn.v gemv_engine.v KV_cache.v | 需回歸 (ddr_top 補 shadow tie 8 腳 + sq_valid=0 等 4 腳) |
| tb_q_frontend | KV_frontend.v RAD_attn.v | 既存 (q_frontend 例化 q_pairmag_unit → 需 RAD_attn.v) |
| tb_rmsnorm | rmsnorm.v | 既存 |

## 備註

- 兩個部署 profile 並存: **片上** (KV_cache 直讀, demo S=2048 單層單 head) 與
  **DDR** (KV_ddr 子系統, 全模型 28L×8KV)。s_k side-plane 兩 profile 皆常駐片上
  (片上走 k_cache 隨行 / DDR 走 sk_ram 獨立)。
- rope_unit 同時服務 Q 路與 K 路 (q_frontend / k_feeder 皆例化; 檔名歸 KV_frontend 係按 K 管線主用途)。
- k_quant_unit v1 (單 buffer) 為 spec dead, 未收錄 (v2 的 bit-exact 對照基準僅存 tb)。
- **RoPE LUT 資料** (theta base=500000 / cos / sin, Q1.15) 由 TB 層次自算或部署 init 引擎載入;
  rope_lut_gen 無自初始化。tb_k_feeder 內建自算 + 假通過防護 (全 0 表 FATAL)。
- **C2 (softmax runtime scale)**: 舊常數 SCALE_MUL 語義在真 decode 為錯 (s_q per-token per-head)。
  SM bank reset 值 = 參數換算 (522/2^12 → mant=1044/exp=-3), 不寫 bank = legacy bit-exact,
  q_frontend 每 token 每 head 寫入覆蓋 (寫時 exp 摺 -SQ_EXP_FOLD=12 消 SFS 記帳差)。
- **C3 (stage1 因果縫接)**: flush-per-step (廢 write-combine) 與 writer stg 1B 讀口 patch
  (~45K cycle/step) 皆斃; ktail_shadow 鏡射 0 額外 cycle 為定案。

## 缺件 (RTL 不存在)

| 件 (Component) | 說明 (Description) |
| :--- | :--- |
| **decode 序列器** | per token × 8 KV-head × (3 Q-head) 頂層縫接: Q/K/V 前端調度 → writer → shadow fence → krunmax load + pairmag 選擇 → step_start → 收 3 attn_out。葉件 (k_feeder/krunmax_bank/sk_ram/SM bank/shadow) 全備並驗, 待收口 |
| **CETT FFN** | 整條 up/gate GEMV → act → CETT 閾值跳過 → down；linear_engine 可重用，CETT 選擇邏輯與 W_down_norm 讀排程沒有 |
| **weight fetch 排程** | AWQ INT4 權重 DDR → linear_engine 的流 + 雙緩衝；addr_gen 只有 W_DOWN 位址 |
| **layer sequencer** | 28 層輪替、per-layer base 換頁、chunk 邊界控制 (含 krunmax_bank clear 的 chunk 邊界觸發) |
| **AXI4 真 shim** | DDR 模型是抽象 1req=1row；真 AR/R/AW/W/B burst master + Zynq DDRC QoS。ktail_shadow 的 B1 (K_row/V 直寫 fence) 亦歸此件 |
| **prefill tile 引擎** | tile 化 QK^T (URAM-bound)；現有件全 decode 導向 |

> **已補齊 (從舊缺件表移除)**: ~~s_k 片上 RAM~~ → sk_ram (§3); ~~SCALE_MUL CSR~~ → softmax_unit runtime SM bank (§5)。
