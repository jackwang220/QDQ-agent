# QDQ Agent — 量化模型轉換工具

將 QAT（量化感知訓練）訓練好的模型自動轉成硬體可部署的量化 ONNX。

```
best.pt + model.txt  →  [QDQ Agent]  →  model_bias_topo.onnx
```

> 快速上手看 [README_user.md](README_user.md)
> 訓練框架說明看 [README_qcraft.md](README_qcraft.md)

---

## 設計動機

原始版本是一組獨立 script，需要手動依序執行、手動修改 code 裡的 hardcode layer index，換模型成本高。

Agent 版本改為：
- 所有步驟由 LangGraph agent 自動串接
- layer index（SPPCSPC、RepConv、Concat 等）從 `model.txt` 自動推斷
- postprocess 設定（transpose、nodes_to_remove）從 ONNX 自動推斷
- Human-in-the-Loop（HITL）讓使用者在關鍵步驟確認或修正
- 換模型只需改 `pipeline_config.yaml`

---

## Pipeline 架構

```
load_config
    │
run_step0  (generate_model_txt — 有 model.txt 則跳過)
    │
load_model_txt → infer_layer_constants
    │
[HITL] review_quantizer_mode
    │
run_step1  (export_model_excel)
    ├─ 有 unknown node → suggest_unknown_fix → [HITL] review_unknown_fix → 回 step1
    └─ 無 unknown node ─┐
                        ▼
              check_detect_layer
              ├─ 已知 → run_step2
              └─ 未知 → infer_detect_layer → [HITL] review_detect_layer → run_step2
                        │
                   run_step3 → run_step4
                        │
              scan_qdq_coverage → [HITL] review_qdq_coverage
                        │
                   run_step5 → infer_postprocess
                        │
              [HITL] review_input_bias
                        │
                   run_step6  (最終輸出)
```

---

## 各步驟說明

| Step | Script | 說明 |
|------|--------|------|
| 0 | `generate_model_txt.py` | 從 `.pt` 生成 `model.txt`（有則跳過） |
| 1 | `export_model_excel.py` | 解析 `model.txt`，產出含 FL 值的 Excel |
| 2 | `export_quant_fused.py` | PyTorch → raw ONNX（有 `*_sim_annotate.onnx` 則跳過） |
| 3 | `model_fp32_int8.py` | weight/bias dtype FP32 → INT8 |
| 4 | `onnx_view_wrap_all_repconv_topo.py` | 根據 Excel FL 值插入 Q/DQ 節點 |
| 5 | `implicit_topo.py` | 處理 implicit 輸出層（IDetect implicit multiply/add） |
| 6 | `modify_model_topo.py` | 後處理：加 input bias、補缺漏 Q/DQ、加 Transpose、清理多餘節點 |

---

## HITL 中斷點說明

| 中斷點 | 時機 | 說明 |
|--------|------|------|
| `quantizer_mode_review` | Step 1 前 | 確認從 model.txt 偵測到的 quantizer 類型（DFPQuantizer 系列） |
| `unknown_fix_review` | Step 1 後（有 unknown node 時） | LLM 建議的 role_type pattern，需手動套用到 `export_model_excel.py` |
| `detect_layer_review` | Step 2 前 | 確認 IDetect 層的 layer index |
| `qdq_coverage_review` | Step 4 後 | 確認缺少 Q/DQ 覆蓋的節點與 FL 值 |
| `input_bias_review` | Step 6 前 | 確認 input_bias 值（推論前加到 model input） |

---

## 環境需求

- GPU server（Step 0、2 需要 PyTorch + GPU 執行 `export_quant_fused.py`）
- 若已有 `best_sim_annotate.onnx`，Step 2 自動跳過，不需要 GPU
- 若已有 `model.txt`，Step 0 自動跳過，不需要 yolov7

```bash
pip install -r requirements_qdq.txt   # pipeline 套件
pip install -r requirements_agent.txt  # agent + web panel 套件
cp .env.template .env                  # 填入 ANTHROPIC_API_KEY
```

---

## yolov7 框架依賴說明

### 什麼是這個 yolov7

這不是 GitHub 上的公開版 [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)，而是實驗室基於它修改的**內部 fork**，加入了量化相關的模組（`quant_model_v7_revised.py`、`QuantModel`、各種 `DFPQuantizer` 等）。這些模組是 QAT 訓練和 ONNX 轉換的核心，**無法從 pip 安裝，需要使用與訓練時相同的本地 repo**。

### 哪些步驟需要它

| Step | 用途 | 缺少時的行為 |
|------|------|-------------|
| Step 0 (`generate_model_txt.py`) | `torch.load(best.pt)` 需要 `QuantModel` 等 class 在 sys.path | 報錯（找不到 class） |
| Step 2 (`export_quant_fused.py`) | 用 yolov7 的 export 流程輸出 `*_sim_annotate.onnx` | 報錯（import `models` 失敗） |

Step 1、3、4、5、6 **完全不需要** yolov7。

### 技術細節

pipeline 在執行這兩個 step 時，會把 `yolov7_dir` 設為 `cwd` 並加入 `sys.path`，讓 `import models` 能正確解析到 yolov7 目錄下的模組：

```python
# nodes/pipeline.py 的做法
yolov7_abs = str(Path(yolov7_dir).resolve())
subprocess.run([sys.executable, script, ...], cwd=yolov7_abs)
```

### 跳過條件（不需要 yolov7 的情況）

```
有 model.txt          → Step 0 自動跳過，不需要 yolov7
有 *_sim_annotate.onnx → Step 2 自動跳過，不需要 yolov7
兩者都有              → 完全不碰 yolov7
```

若 server 上已有 `best_sim_annotate.onnx` 和 `model.txt`，正常使用時 Step 0、2 都會跳過，完全不需要 yolov7。換到新環境時才需要重新準備。

---

## 設定檔 pipeline_config.yaml

```yaml
model:
  weights: "./best.pt"      # QAT 訓練好的 .pt 路徑
  img_size: 640             # 模型輸入大小
  detect_layer: null        # IDetect layer index；null = agent 自動推斷
  input_bias: -0.5          # 訓練時有做 /255 再 -0.5 平移 → -0.5；只做 /255 → 0.0

paths:
  code_dir: "./code"
  yolov7_dir: "/path/to/yolov7"
  model_txt: "./model.txt"  # "auto" = 從 weights 自動生成
  output_dir: "./wrap_all_model_onnx_export"
  excel_output: "./model_fl_values_fixed.xlsx"

intermediate:               # 中間檔名，通常不用改
  raw_onnx: "wrap_all_best_temp.onnx"
  int8_onnx: "int8_converted_model.onnx"
  qdq_onnx: "wrap_all_temp.onnx"
  implicit_onnx: "output_topo.onnx"
  final_onnx: "model_bias_topo.onnx"

postprocess:                # 不用預先填，agent 自動推斷並透過 HITL 確認
  extra_qdq_nodes: []
  transpose_configs: []
  outputs_to_remove: []
  nodes_to_remove: []
```

---

## 啟動方式

### Web Panel

```bash
python -m qdq_agent.server --host 0.0.0.0 --port 8000
```

瀏覽器開 `http://<server-ip>:8000`。

從外部 SSH 連進 server 時，可用 port forwarding 在本機開：
```bash
ssh -L 8000:localhost:8000 user@your-server
# 然後開 http://localhost:8000
```

### CLI

```bash
python -m qdq_agent.main --config pipeline_config.yaml
python -m qdq_agent.main --config pipeline_config.yaml --thread-id run_02
```

---

## 輸出

```
wrap_all_model_onnx_export/
├── wrap_all_best_temp.onnx    Step 2 raw ONNX
├── int8_converted_model.onnx  Step 3 weight dtype 轉 INT8
├── wrap_all_temp.onnx         Step 4 插入 Q/DQ
├── output_topo.onnx           Step 5 處理 implicit 輸出層
└── model_bias_topo.onnx       ★ 最終結果
```

---

## 專案結構

```
QDQ-agent/
├── pipeline_config.yaml       設定檔
├── code/                      各步驟 script（原始交接版本）
│   ├── generate_model_txt.py
│   ├── export_model_excel.py
│   ├── export_quant_fused.py
│   ├── model_fp32_int8.py
│   ├── onnx_view_wrap_all_repconv_topo.py
│   ├── implicit_topo.py
│   └── modify_model_topo.py
├── qdq_agent/
│   ├── graph.py               LangGraph 圖定義（節點與邊）
│   ├── state.py               QDQState TypedDict
│   ├── llm.py                 LLM client + tracing
│   ├── main.py                CLI 入口
│   ├── nodes/
│   │   ├── pipeline.py        各 step 的執行節點
│   │   └── agent.py           LLM 推斷 + HITL 中斷節點
│   ├── prompts/               LLM prompt 模板
│   └── server/                Web Panel（FastAPI + WebSocket）
│       ├── app.py
│       ├── panel.py           前端 HTML（vanilla JS，no framework）
│       ├── session.py         pipeline thread ↔ asyncio bridge
│       └── __main__.py
├── requirements_qdq.txt
└── requirements_agent.txt
```

---

## 移植到新模型的注意事項

換一個架構不同的模型時需要留意：

1. **layer index 推斷失敗**：agent 用 regex 從 `model.txt` 推斷，若模型類型不同（非 YOLOv7 系列）可能推錯，HITL 時要仔細確認
2. **unknown role_type**：Step 1 Excel 出現 `Unknown` 時，表示 `export_model_excel.py` 的 path 規則沒有覆蓋到新的層類型，需手動在該 script 裡新增對應條件
3. **新的 wrapper 類型**：若 `quant_model_v7_revised.py` 有新增 wrapper，`export_model_excel.py` 和 `onnx_view_wrap_all_repconv_topo.py` 的 node_type 對應表也要同步更新
4. **輸出層結構不同**：Step 6 的 transpose 和 nodes_to_remove 由 agent 自動推斷，若偵測層結構異常需手動確認

---

## 環境建議

| 項目 | 說明 |
|------|------|
| Python | 3.10 以上 |
| CUDA | Step 0、2 需要 GPU；其餘步驟 CPU 即可 |
| yolov7 路徑 | 本地 repo 根目錄，填入 `pipeline_config.yaml` 的 `yolov7_dir` |
| conda 環境 | 建議用獨立環境，避免套件衝突 |
