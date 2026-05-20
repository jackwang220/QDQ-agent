# QDQ Agent — YOLOv7 量化模型轉換工具

將 QAT（量化感知訓練）訓練好的 YOLOv7 模型自動轉成硬體可部署的量化 ONNX。

```
best.pt + model.txt  →  [QDQ Agent]  →  model_bias_topo.onnx
```

> 快速上手版請看 [README_user.md](README_user.md)
> 原始交接文件請看 [README_qcraft.md](README_qcraft.md)

---

## 背景與設計動機

原始版本（`QDQ交接/QDQ交接/` 資料夾）是一組獨立的 Python script，需要手動依序執行、手動修改 code 裡的 hardcode layer index，換模型成本高。

Agent 版本改為：
- 所有步驟由 LangGraph agent 自動串接
- layer index（SPPCSPC、RepConv、Concat 等）從 `model.txt` 自動推斷
- postprocess 設定（transpose、nodes_to_remove）從 ONNX 自動推斷
- 換模型只需改 `pipeline_config.yaml` 一個檔案

---

## 需要準備的東西

| 檔案 | 說明 |
|------|------|
| `best.pt` | QAT 訓練好的模型 checkpoint |
| `model.txt` | 訓練時的模型架構與 FL 值，從訓練結果的 `opt_after-best.txt` 複製而來 |

**為什麼需要 `model.txt`**：FL 值（fractional length，決定量化 scale）必須和 `.pt` 訓練時完全一致。雖然可以從 `.pt` 自動生成，但因為 `quant_model_v7_revised.py` 版本差異，自動生成的 FL 值可能不準確，建議優先使用訓練時存下來的版本。

---

## Pipeline 各步驟說明

```
Step 0  generate_model_txt.py          從 .pt 自動生成 model.txt（有 model.txt 則跳過）
Step 1  export_model_excel.py          解析 model.txt，產出含 FL 值的 Excel
Step 2  export_quant_fused.py          PyTorch → raw ONNX（unwrap quantizer，weight 還原為整數）
Step 3  model_fp32_int8.py             ONNX 中的 weight/bias dtype FP32 → INT8
Step 4  onnx_view_wrap_all_repconv_topo.py   根據 Excel FL 值插入 Q/DQ 節點
Step 5  implicit_topo.py               處理 implicit 輸出層
Step 6  modify_model_topo.py           後處理：加 input bias、補缺漏 Q/DQ、加 Transpose、清理多餘節點
```

Agent 在 Step 1 之前自動推斷 layer index，在 Step 4 之後自動掃描 Q/DQ 覆蓋率，在 Step 5 之後自動推斷 postprocess 設定。有兩個 Human-in-the-Loop 中斷點讓使用者確認 detect layer 與補插的 Q/DQ 節點。

---

## 環境設定

**需求**：GPU server（Step 0、2 需要 PyTorch + GPU）

```bash
# 1. 安裝 pipeline 套件（需要在 GPU server 上）
pip install -r requirements_qdq.txt

# 2. 安裝 agent 套件
pip install -r requirements_agent.txt

# 3. 設定 API key
cp .env.template .env
# 編輯 .env，填入 ANTHROPIC_API_KEY
```

---

## 設定檔 pipeline_config.yaml

換模型只需要改以下欄位，其餘不動：

```yaml
model:
  weights: "./best.pt"      # QAT 訓練好的 .pt 路徑
  img_size: 640             # 模型輸入大小
  detect_layer: null        # IDetect layer index；null 讓 agent 自動推斷
  input_bias: -0.5          # 輸入偏移，DLA 硬體用 -0.5，不需要則設 0.0

paths:
  yolov7_dir: "/path/to/yolov7"   # YOLOv7 repo root
  model_txt: "auto"               # 有 model.txt 就填路徑；沒有填 "auto" 自動生成
  output_dir: "./wrap_all_model_onnx_export"
```

`postprocess` 區塊（extra_qdq_nodes、transpose_configs、nodes_to_remove 等）**不需要預先填寫**，agent 會在執行時自動推斷並透過 HITL 讓你確認。

---

## 執行

```bash
cd ~/QDQ-agent
python -m qdq_agent.main --config pipeline_config.yaml

# 指定 thread id（重跑時避免 checkpoint 衝突）
python -m qdq_agent.main --config pipeline_config.yaml --thread-id my_run_02
```

執行過程會有兩個需要按 Enter 確認的中斷點：

1. **detect_layer 確認**：agent 推斷出 IDetect layer index，確認或輸入其他數字
2. **Q/DQ 覆蓋率確認**：列出缺少 Q/DQ 的節點與推斷的 FL 值，確認或逐一覆蓋

正常情況直接按 Enter 接受即可，全程約 15–30 秒。

---

## 輸出

```
wrap_all_model_onnx_export/
├── wrap_all_best_temp.onnx    Step 2 輸出，raw ONNX
├── int8_converted_model.onnx  Step 3 輸出，weight dtype 轉 INT8
├── wrap_all_temp.onnx         Step 4 輸出，插入 Q/DQ 後
├── output_topo.onnx           Step 5 輸出，處理 implicit 後
└── model_bias_topo.onnx       ★ 最終結果，給硬體部署用
```

---

## 如何取得正確的 model.txt

訓練 script `train_quant_v7.py` 訓練完成後，會在 `runs/train/expXX/opt_after-best.txt` 存下 `str(model)` 的內容。將這份檔案複製並改名為 `model.txt` 即可。

```bash
cp runs/train/expXX/opt_after-best.txt ~/QDQ-agent/model.txt
```

如果想讓訓練 script 自動存到固定位置，可在 `train_quant_v7.py` 的 `line 623` 後加：

```python
with open(wdir / "model.txt", "w") as f:
    f.write(str(model))
```

這樣 `weights/best.pt` 和 `weights/model.txt` 就會自動成對存在。

---

## 各 script 功能說明

| 檔案 | 對應步驟 | 說明 |
|------|----------|------|
| `code/generate_model_txt.py` | Step 0 | 從 `.pt` 生成 `model.txt` |
| `code/export_model_excel.py` | Step 1 | 解析 `model.txt` 產出 FL 值 Excel；支援 `--concat-layers`、`--maxpool-layers`、`--upsample-layers` 覆蓋預設 |
| `code/export_quant_fused.py` | Step 2 | PyTorch → ONNX，weight 輸出整數值 |
| `code/model_fp32_int8.py` | Step 3 | ONNX weight dtype FP32 → INT8 |
| `code/onnx_view_wrap_all_repconv_topo.py` | Step 4 | 插入 Q/DQ 節點；支援 `--sppcspc-layer`、`--repconv-layers`、`--detection-layer` 覆蓋預設 |
| `code/implicit_topo.py` | Step 5 | 處理 implicit 輸出層 |
| `code/modify_model_topo.py` | Step 6 | 後處理（bias、補 Q/DQ、Transpose、清理節點） |
| `qdq_agent/` | 全程 | LangGraph agent，自動推斷參數並串接所有步驟 |

---

## 新模型支援注意事項

換一個結構不同的模型時，agent 的自動推斷可能不完整，需要留意：

1. **layer index 推斷失敗**：agent 從 `model.txt` 用 regex 推斷，若模型結構差異大可能推錯，HITL 時要仔細確認
2. **unknown role_type**：Step 1 產出的 Excel 中若出現 `role_type = unknown` 的節點，表示 `onnx_view_wrap_all_repconv_topo.py` 裡沒有對應的字串處理邏輯，需要手動補上
3. **新的 wrapper 類型**：若 `quant_model_v7_revised.py` 有新增 wrapper，`export_model_excel.py` 和 `onnx_view_wrap_all_repconv_topo.py` 裡的 node_type 對應表也要跟著補
4. **輸出層結構不同**：Step 6 的 transpose 和 nodes_to_remove 由 agent 自動推斷，但若偵測層結構異常，需要手動確認輸出點名稱

---

## Server 資訊（實驗室）

| 項目 | 值 |
|------|-----|
| IP | 140.113.228.206 |
| 帳號 | m314832018 |
| yolov7 路徑 | `/home1/m314832018/yolov7` |
| QDQ-agent 路徑 | `~/QDQ-agent` |
| Python 環境 | `~/miniforge3/envs/ivs` |
