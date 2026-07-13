# RTL 檔案結構 Spec (整併版, 15 檔 → 6 檔)

整併原則: **零改碼** — 各來源檔逐字保留為段落 (含各自 header/timescale)，僅加檔頭與
`// ===== 來源: xxx.v =====` 分隔。行尾正規化為 LF。模組名全域唯一，六檔可聯編。

---

## 1. KV_frontend.v — 量化 / RoPE 前端子系統

| 模組 | 來源 | 角色 |
|---|---|---|
| `rope_lut_gen` | rope.v | RoPE cos/sin LUT 生成 |
| `rope_unit` | rope.v | 逐對旋轉 (Q.16)，K 路與 Q 路共用 |
| `recip_seed_nr` | K_frontend.v | 倒數: seed LUT (256×16b) + 1 次 Newton-Raphson; LZD e/m 分解一物三用 (recip 尾數索引 / e_sk / m_sk)。**K 與 V 量化共用** |
| `k_quant_unit_v2` | K_frontend.v | K per-token INT8 量化 (scale=absmax/128)，ping-pong 雙 buffer 配 RoPE 收流；輸出 k_int8 / sign plane / s_k (mant+exp) |
| `k_pipeline_top` | K_frontend.v | rope→quant 串接頂層 + runmax 併流 (供 RAD anchor) |
| `v_quant_unit` | V_quant_unit.v | V per-token KIVI INT4 (G=32): min/max 掃 → 15/range 倒數路 → vq/m_s/e_s/m_q412；range==0 合法路；s 分解走全精度 range×4369 再 LZD |

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
| `kdim_prefetch` | kdim_prefetch.v | 定案 (a) stage1 Active-K 預填: pair_mask → K_DIM/K_SIGN 條帶 session (idx 即位址) → dim-slot banks → 1-cycle 合成 row (active←真值, dropped←sign±1)；3Q 共用一次 fill |
| `gqa_union` | Gqa_union.v | GQA 3:1 查詢側歸屬: 3 Q-head Top-N → 去重 union (bitmap+epoch, 回卷掃清) + mask 查詢 + qpos 映射；drain 直驅 kv_rd_sched |

## 4. RAD_attn.v — RAD 注意力核心 (原樣未動)

| 模組 | 角色 |
|---|---|
| `topk_engine` | 串流 Top-N 選擇 (cand_idx) |
| `qk_pu` | Q·K 對級 PU: acc_act (active) + acc_sign (dropped, Σ\|q\|·sign(k)) 雙累加 |
| `score_fixup` | F1 定點: act·m_sk·2^(e_sk+F_S-10) + (acc_sign>>>β_sh)<<<F_S → Q.F_S=6 |
| `krunmax_unit` | K running-max (anchor 選擇源) |
| `q_pairmag_unit` | Q pair 幅值 (dynamic 選擇源) |
| `attn_score_core` | stage1 全掃 + Top-N + stage2 全精度重算；**內建 GQA 組迭代 (g_q) 與候選選擇**；req/data 串流介面 (按序)；mode=1 硬截斷 / mode=0 soft-tail |

## 5. attn_chain.v — 注意力消費鏈子系統

| 模組 | 來源 | 角色 |
|---|---|---|
| `softmax_unit` | softmax.v | 定點 base-2 softmax: SCALE_MUL **runtime 校準** (摺 s_q 與 F_S；Q.6 值域→522/64≈8)，FP16 aw (mant=0 精確標記 aw=0) |
| `attn_chain_top` | Attn_chain_top.v | decode 消費鏈頂層 (**片上 profile**): score core ↔ k_cache portC 橋 (+Q bank) → softmax → **aw≠0 過濾** V feeder → v_engine；consume_k/v_done 脈衝 (kv_rd_sched 口徑) |
| `attn_chain_ddr_top` | attn_chain_ddr.v | decode 消費鏈頂層 (**DDR profile**): K 路 = kdim_prefetch+bridge+gather#K (3 DDR master, AXI interconnect 假設)；V 路 = awb list → gather#V go_v → v_engine；attn_out 與片上版 **bit-exact** (tb_attn_chain_ddr 雙 profile 對拍) |

## 6. gemv_engine.v — GEMV / attn·V 引擎 (原樣未動)

| 模組 | 角色 |
|---|---|
| `gemv_lane` | INT 乘加 lane |
| `psum_premul_acc` | 部分和預乘累加 |
| `linear_engine` | 投影 GEMV (輸出 Q.16, 24b, 整向量 pulse — v_quant 輸入源) |
| `v_engine` | attn·V: Σ(aw·s)·Vq + Σaw·m 係數摺疊，Q3.20 輸出 |
| `fp_normalize` | Q3.20 → FP16 |

---

## 資料流 (decode step)

```
[KV_frontend]  rope → k_quant ─┬→ [KV_cache] k_cache      (片上 profile)
                               └→ [KV_ddr]   kv_ddr_writer (DDR profile)
[gemv_engine]  linear_engine → [KV_frontend] v_quant ─┬→ v_cache / kv_ddr_writer

[RAD_attn] attn_score_core ──req──> k_cache portC (片上)
                            └─req──> [KV_ddr] kv_req_bridge → kv_gather_top (DDR)
           f 流 → [attn_chain] softmax → aw 過濾 → v_cache / gather V phase
                                        → [gemv_engine] v_engine → attn_out
多 head: [KV_ddr] gqa_union (3Q→union) → kv_rd_sched (per-KV-head 輪詢)
```

## TB → 檔案對照 (整併後編譯清單)

| TB | 編譯來源 |
|---|---|
| tb_k_cache / tb_v_cache | KV_cache.v |
| tb_integ_kcache | KV_cache.v KV_frontend.v |
| tb_v_quant | KV_frontend.v |
| tb_integ_vchain | KV_frontend.v KV_cache.v gemv_engine.v |
| tb_kv_gather / tb_kv_ddr_writer / tb_kv_writer_mh / tb_kv_rd_sched / tb_gqa_union / tb_kv_req_bridge | KV_ddr.v |
| tb_attn_chain | attn_chain.v RAD_attn.v gemv_engine.v KV_cache.v KV_frontend.v |
| tb_stage1_equiv | KV_ddr.v RAD_attn.v |
| tb_attn_chain_ddr | attn_chain.v KV_ddr.v RAD_attn.v gemv_engine.v KV_cache.v |

## 備註

- 兩個部署 profile 並存: **片上** (KV_cache 直讀, demo S=2048 單層單 head) 與
  **DDR** (KV_ddr 子系統, 全模型 28L×8KV)。s_k side-plane 兩 profile 皆常駐片上。
- rope_unit 同時服務 Q 路 (檔名歸 KV_frontend 係按 K 管線主用途)。
- k_quant_unit v1 (單 buffer) 為 spec dead, 未收錄 (v2 的 bit-exact 對照基準僅存 tb)。

## 缺件 (RTL 不存在)
# 硬體加速器模組與狀態盤點

| 件 (Component) | 說明 (Description) |
| :--- | :--- |
| **CETT FFN** | 整條 up/gate GEMV → act → CETT 閾值跳過 → down；linear_engine 可重用，CETT 選擇邏輯與 W_down_norm 讀排程沒有 |
| **weight fetch 排程** | AWQ INT4 權重 DDR → linear_engine 的流 + 雙緩衝；addr_gen 只有 W_DOWN 位址 |
| **layer sequencer** | 28 層輪替、per-layer base 換頁、chunk 邊界控制 |
| **s_k 片上 RAM (獨立)** | DDR profile 需獨立 BRAM 模組 (寫口接 k_pipeline)；現在只活在 k_cache 內 |
| **AXI4 真 shim** | DDR 模型是抽象 1req=1row；真 AR/R/AW/W/B burst master + Zynq DDRC QoS 設定 |
| **prefill tile 引擎** | tile 化 QK^T (URAM-bound)；現有件全 decode 導向 |
| **SCALE_MUL CSR** | softmax 常數目前是 elaboration 參數，部署要 per-row s_q runtime 寫 |
