# Llama 3B FPGA RTL 規格（現行）

更新：2026-07-20  
目標器件：AMD/Xilinx Kria KV260（Zynq UltraScale+ MPSoC）  
模型基準：Llama 3.2 3B，`D_HEAD=128`、`N_Q=24`、`N_KV=8`、GQA `3:1`、`D_FF=8192`

本文件只描述目前仍有效的硬體契約。舊版把 `S_MAX=2048`、`K_MAX=64`、稠密
`cand_mask/map_r` 視為長期架構的內容已刪除；這些只能作為 legacy mixed-profile
回歸基準，不可作為 `S>=8192` 的綜合組態。

## 1. 系統定案

### 1.1 單一 bitstream

同一個 request 會先 prefill 再 decode，因此不得以重新下載 bitstream 作為正常
mode 切換。最終資料路徑在同一 bitstream 內共用 QK、V、量化、DDR gather 與
GEMV lane，runtime 選擇：

- `PREFILL_ONLINE`：分 tile 的 online/FlashAttention recurrence。
- `DECODE_SPARSE`：全 S 串流 stage1 Top-N，bounded union，候選 full-score，
  128-depth softmax。
- `DENSE_FALLBACK`：Layer-0 或 CETT keep-all/資源溢位時保持演算法正確。

mode mux 必須位於已寄存的模組邊界，不可把 128K address、384-way candidate 或
8192-way neuron 判斷合成成大扇入組合 mux。

Union engine 也必須 runtime 切換：

- Short profile（目前 `S<=2048`）：保留 bounded short-mask 枚舉，延遲較低。
- Long decode：使用 hash union，容量不隨 S 成長。

Hash 每個 candidate 約需 read/check 三拍，另有 table clear；短 S 不一定比
mask 快。若為省面積只保留 hash，必須明列為面積換延遲。以 384 candidates
粗估，mask 掃描延遲約在 S 接近 70K 後才可能超過 hash；精確 crossover 以
合成後 cycle profile 為準。

### 1.2 長上下文位寬

部署 profile 以 128K context 為上限：

| 欄位 | 位寬 | 說明 |
|---|---:|---|
| token index | 17 | `0..131071` |
| sequence length/count (`SW`) | 18 | 必須表示 `131072`；不可只用 17-bit |
| RoPE position (`POS_W`) | 17 | index 不需要表示 length |
| writer/shadow count (`TIDX_W`) | 18 | `n_tok`、`stg_base+stg_cnt` 不溢位 |
| Top-N (`NW`) | 8 | 可表示 128 |
| `K_MAX` | 128 | `S>=8192` 使用 `N=128` |
| candidate union | 最大 384 | `3*K_MAX`，與 S 無關 |
| DDR byte address | 32 | base/offset 加法須在 32-bit 域完成 |

legacy 頂層目前仍以 `SW=12/S_MAX=2048/K_MAX=64` 作 bit-exact 回歸。長上下文
整合時必須一次傳遞上述 profile；只改外層 `S_MAX` 不算完成。

## 2. 長上下文 attention

### 2.1 Decode 資料流

1. Stage1 依 token index `0..S-1` 順序掃描 DDR。
2. 三個 Q head 同拍計算近似分數，只更新各自 depth-128 Top-N；不得寫
   `s1_buf[S]`。
3. 三份 candidate index list 送入 bounded union。不得使用 `cand_mask[S]`。
4. Union 先依 token index 做 sequential radix sort，payload 保留原 union slot。
5. K/V gather 依排序結果發 request；返回資料仍寫回 payload slot。
6. Stage2 僅對 union（最多 384）算 full score。
7. 每個 Q head 僅把自己的 128 個 full score 送入 softmax，再作 `attn*V`。

此流程中，片上 score/candidate/softmax 容量只跟 `K_MAX` 或 `3*K_MAX` 有關，
不跟 S 成長。

Top-N canonical ordering 固定為：

1. signed score 較大者優先。
2. score 相同時 token index 較小者優先。

到達順序不得參與 tie-break。`topk_engine` 使用獨立 `cell_vld`；不能以
`NEG_INF` 同時表示空槽，否則真實飽和 score=`NEG_INF` 會漏選。
`Tb_stage1_top128_equiv.v` 是移除 `s1_buf[S]` 前的硬 gate：full-store reference
先保存全部輸入，再獨立 selection；DUT 為 streaming Top-128。

### 2.2 Candidate union 與 `map_r`

已實作 `gqa_sparse_union_bram`：

- 1024-entry power-of-two hash table。
- Registered read/check；collision 逐拍 linear probe。
- 384-entry union RAM、candidate `{key,slot}` RAM。
- 每 step 確定性清 1024 entry，沒有 epoch wraparound 假命中。
- 不使用 S-bit mask、S/64 掃描或 384-way CAM。

已實作 `candidate_radix_scheduler`：

- 預設 radix-16，17-bit key 共 5 pass。
- Histogram/prefix/scatter 均為 sequential RAM 操作。
- 輸出 `{sorted_key, original_union_slot}`，所以 reorder 不改變 head ownership。
- 排序本身不會減少 AR 數。是否改善 DDR locality 必須用真實 candidate trace
  驗證，不能由演算法結構推定。

`analyze_candidate_locality.ps1` 接受每行一組逗號分隔 candidate index，輸出
排序前後平均距離、4-KiB page crossing，以及「若相鄰 row 可 burst」的 AR
減少率。取得 LongBench 真實 Top-N trace 前，radix scheduler 的效能收益狀態
為「未證實」；它目前只提供確定性 ordering 與 future coalescer 的前置條件。

舊 `Attn_chain.v::map_r[0:S_MAX-1]` 是隱藏的 S-linear state，現已由
`bounded_sorted_lookup` 取代。它在最多 384 個嚴格升序 union key 上作 registered
binary search；沒有 128K asynchronous RAM 或大 CAM。單元 gate 已通過，
跨 `T_BLK` end-to-end checksum 尚待 ModelSim gate。

### 2.3 DDR gather 碎讀

Union 最壞 384 個 token，原始 Top-K 順序可能造成 random-read 風暴。定案：

- 先由 radix scheduler 依 token index 排序。
- gather 保留至少 16 個 outstanding request 與 tag reorder。
- 是否有 DDR page locality 收益由真實 trace 決定；相鄰 token row 才可由 AXI
  adapter 合併。
- 目前 RTL read master 只有 `ar_addr/ar_tag`，沒有 `ar_len`。因此真正多-beat
  burst coalescing 尚未接線；在新增 `ar_len` 前，不得宣稱已消除碎讀。

Stage1 是順序 dim-plane/tiled stream；random gather 只發生在 bounded stage2/V
候選，不會成為 O(S) 次 random read。

Burst-capable read contract v2 已固定於 `Axi_read_contract.vh`：

| 訊號 | 契約 |
|---|---|
| `ar_len[7:0]` | beats minus one；legacy 單 beat為 0 |
| `ar_tag` | 一個 burst 一個 tag，直到 `r_last` 前不得重用 |
| `r_beat[7:0]` | burst 內 zero-based beat |
| `r_last` | `r_beat==ar_len` 的 handshake |
| outstanding | 預設 16，且不得超過 `2**TAG_W` |
| boundary | coalesced burst 不跨 4-KiB |

目前 `gather_fsm` 是 v1，等價 `ar_len=0/r_last=1`。v2 尚未接入，不能因為已有
radix sorting 就宣稱 DDR AR 數或 burst bandwidth 已改善。

### 2.4 Softmax

`softmax_unit` 已把 token-domain `S_MAX` 與物理 `BUF_MAX` 分離：

- Long decode：`BUF_MAX=K_MAX=128`。
- Legacy dense prefill：`BUF_MAX=S_MAX`。
- `SUMW` 由 `BUF_MAX` 自動導出。
- `n_f==0` 或 `n_f>BUF_MAX` simulation fail-loud。

Long decode 不對 128K 個分數作 softmax。Stage1 已完成 global Top-N，stage2
softmax 的輸入最多 128，因此仍可先找 max，再作既有 bit-exact base-2 softmax。

Long prefill 與 Layer-0 dense 才使用 online recurrence。每 tile 更新：

```text
m' = max(m, max(tile_score))
l' = l * exp(m-m') + sum(exp(tile_score-m'))
o' = o * exp(m-m') + sum(exp(tile_score-m') * V_tile)
```

最終輸出 `o/l`。實作必須使用現有 base-2 LUT/reciprocal 定點域，不能加入 FP32
divider。online prefill 尚未接入目前 RTL，列為整合 gate。

### 2.5 K prefetch、`s_k` 與 ktail

`kdim_prefetch` 已移除下列 2048-only 寫死位寬：

- active word address：固定 8-bit → `$clog2(S_MAX/8)`。
- active segment：固定 5-bit → 由 `N_ASEG` 導出。
- sign word-pair address：固定 4-bit → 由 `SWRD` 導出。
- 非整除 segment/word 數改為 ceil。

這使 `S_MAX=8192` 不再因內部地址截斷而 alias；但把整個 128K prefetch buffer
放片上仍不符合 KV260 資源限制。最終 long profile 必須用雙 buffer rolling tile
（建議每 bank 2048 token）：

- producer 只可領先 consumer 一個 tile，避免覆寫未消費資料。
- active byte 與 dropped-sign plane 使用相同 tile epoch。
- tile base 與 local offset 分開寄存，BRAM address 只使用 local offset。
- bridge 在 `tile_base <= key < tile_limit` 時服務，否則 backpressure。

`ktail_shadow` 的資料容量只跟 `HKV*T_BLK*D` 有關；`T_BLK=64` 不隨 S 增長。
長 context 只需把 `n_tok/stg_base/stg_cnt/window_end` 加寬至 18-bit，並保留
`wr_valid && wr_ready` 同拍 commit 契約。

`sk_ram[HKV][S]` 也是隱藏 S-linear state。128K profile 應把 `s_k` metadata
寫入 DDR side-plane，並隨 K tile/gather 帶回；不可配置成 8×128K×17-bit
片上 RAM。

## 3. CETT FFN

### 3.1 演算法契約

Gate/up projection 與 SwiGLU 必須完整計算：

```text
h_i = SiLU(gate_i) * up_i
keep_i = (abs(h_i) * norm_i >= epsilon)
norm_i = ||W_down[:, i]||
```

CETT 只跳過 dropped neuron 對 `W_down` 的 column read/MAC，不可跳過 gate/up，
也不可把多組 scale 平均成單一 scale。

`W_down` norm 採 per-group-128：

- `norm_code`: unsigned INT8。
- `norm_scale`: normalized 11-bit mantissa + signed exponent。
- epsilon：normalized mantissa + signed exponent。

### 3.2 `cett_selector`

已實作：

- 一拍 `abs(h)*norm_code`。
- 下一拍乘 `norm_scale_mant`。
- 第三拍 normalize exponent/mantissa 並比較 epsilon。
- 三級 elastic valid/ready，完整支援 output backpressure。
- 無 divider、無三乘數串接、無 8192-way reduction。
- 輸出保留 `{idx,h,group_last,keep}`。

比較在整數 mantissa/exponent 域完成；threshold equality 使用 keep，符合
`>= epsilon`。

### 3.3 `cett_down_tile_engine`

已實作 128-output tile：

- kept neuron 的 W_down row fragment 逐拍流入，128 lane 平行累加。
- 每個 128-neuron AWQ group 結束時只套用一次 group scale。
- empty group 合法且貢獻精確為零。
- 64 group 對應 `D_FF=8192`。
- output valid 在 backpressure 下保持。

建議部署排程：

1. gate/up/SwiGLU 順序產生 8192 個 `h_i`。
2. selector 建立自然依 neuron index 排序的 keep list 與 per-group count。
3. 對 hidden output 的 24 個 128-wide tile 逐一重播 keep list。
4. 每個 kept neuron 只讀該 output tile 的 W_down fragment。
5. group count 為零時直接送 group-end，不發 DDR weight request。

若 keep-list 容量設限，overflow 必須切到 dense W_down fallback；不得截掉
neuron，否則違反演算法。

### 3.4 尚未完成的 CETT 接線

- gate/up + SwiGLU producer 與 activation BRAM。
- W_down tile address 中的 output-tile offset。
- norm/epsilon loader。
- keep-list replay controller 與 DDR gather。
- CETT enable/threshold 的 layer runtime register。

`addr_gen` 現有 `W_DOWN` 只算 neuron column 起點，尚未表示 3072-output 的
128-wide tile offset；此介面補齊前，不能宣稱 CETT FFN end-to-end 完成。

## 4. RTL 檔案職責

| 檔案 | 現行職責 |
|---|---|
| `Kv_frontend.v` | Q/K RoPE、量化、V 量化 |
| `Kv_cache.v` | legacy on-chip KV/reference path |
| `Kv_ddr.v` | address/gather/writer/prefetch/shadow/legacy `sk_ram` |
| `Rad_attn.v` | RAD score、Top-K、legacy score core |
| `Attn_chain.v` | softmax、K/V consume chain |
| `Gemv_engine.v` | dense linear/V GEMV primitives |
| `Decode_top.v` | legacy mixed-profile decode sequencer |
| `Rmsnorm.v` | RMSNorm |
| `Long_ctx.v` | bounded GQA union、radix DDR scheduler |
| `Ffn_cett.v` | CETT selector、sparse W_down tile engine |

## 5. 驗證狀態

2026-07-20 本地 Icarus (`-g2012`)：

| Testbench | 結果 | 驗證目的 |
|---|---|---|
| `Tb_long_ctx.v` | PASS | 三頭重複 candidate、union length、key→slot |
| `Tb_candidate_sched.v` | PASS | radix ordering、payload slot、output stall |
| `Tb_cett.v` | PASS | threshold equality/drop、pipeline stall、empty group、W_down output hold |
| `Tb_cett_empty_groups.v` | PASS | 全空、頭空、間隔空、連續空、尾空 |
| `Tb_stage1_top128_equiv.v` | PASS | full-store vs streaming Top-128、tie 與 `NEG_INF` |
| `Tb_bounded_lookup.v` | PASS | sorted key 命中、首尾 slot、各類 miss |
| `Tb_softmax_long.v` | PASS | 128-depth buffer 搭配 128K token key domain |
| `Tb_prefetch_8192.v` | PASS | S=8192 active/sign/segment address 不截斷 |
| `Tb_softmax_p2.v` | PASS | sparse/dense `n_f`、key 隨行、backpressure |
| `Tb_score_ls.v` | PASS（8 cases） | 新 score core 對 frozen gold bit-exact |
| `Tb_ktail_shadow.v` | PASS（10 cases） | staging window/shadow/prefetch bit-exact |

Streaming Top-128 ModelSim/Icarus 對拍 checksum：

| case | checksum |
|---|---|
| tie-heavy-small | `4d53eb9f1839a740` |
| boundary-8191 | `03661ff416732595` |
| all-tie-permuted-8192 | `a940f920e23a14a5` |
| boundary-8193 | `03661ff416732595` |
| real-neg-inf | `ae91fa89e0f6f7f2` |

`Tb_attn_chain_ddr.v` 在本次 Icarus elaboration/run 超過 240 秒，沒有取得完成
結果，因此不列 PASS。先前 `Tb_xtblk.v` 的 tok0..67 bit-exact 是 writer/shadow
修正基線；長 context 新 block 尚未接入該 end-to-end test。

## 6. 整合 Gate（不得跳過）

Long-context phase 只有在下列項目全部完成後才可標示 end-to-end：

- 抽換順序固定：`map_r→bounded map`、`cand_mask→sparse list`、
  `s1_buf→streaming Top-128`；一次只換一件。
- 每次抽換後重跑跨 `T_BLK=64` 的 `Tb_xtblk`，bit-exact、`cks`、`xbits`
  都必須與上一 checkpoint 相同。單元 PASS 不等於整合 PASS。
- legacy `s1_buf/full_buf/cand_mask/map_r` 從 long decode elaboration 移除。
- `gqa_sparse_union_bram` 與 radix scheduler 接入 score/V gather。
- `S_MAX/SW/POS_W/TIDX_W/cfg_s/stride` 以 128K profile 一次加寬。
- prefetch 改 rolling tile；`s_k` 改 DDR/tile metadata。
- prefill/Layer-0 online attention 接線，runtime mode 不換 bitstream。
- W_down output-tile address、keep replay、dense fallback 接線。
- `S=8191/8192/8193`、`S=131071/131072`、union=384、AXI backpressure、
  hash collision、CETT keep-none/keep-all 皆有 directed test。
- 完整 `Tb_xtblk` 與 synthesis RAM/DSP/utilization/timing report 通過。

### 6.1 常駐 S-linear 洩漏表

`check_s_linear.ps1` 每次 RTL 修改後執行。出現新的或改名的 S-dependent RAM
會直接 FAIL，必須先更新本表與架構理由。

| 檔案/陣列 | 現況 | Long profile |
|---|---|---|
| `Rad_attn.s1_buf[0:2]` | legacy S-linear | 必須移除 |
| `Rad_attn.full_buf[0:2]` | legacy S-linear | 改 384-depth slot RAM |
| `Rad_attn.cand_mask[0:2]` | legacy S-bit | 改 sparse candidate list |
| `Attn_chain.map_r` | 已移除 | bounded sorted lookup；待跨 T_BLK gate |
| `Kv_ddr.gqa_union.mram/upos_ram` | legacy S-linear | long path不 elaboration |
| `Kv_ddr.sk_ram` | `HKV*S` | DDR metadata + tile cache |
| `Kv_ddr.kdim_prefetch` active/sign banks | depth-derived | rolling tile，不能 128K 全留 |
| `Kv_cache` K/V arrays | on-chip reference profile | long path不 elaboration |

Testbench 的 full-store reference 可以 S-linear；檢查器只限制可綜合 RTL。
