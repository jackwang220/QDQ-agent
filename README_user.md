# QDQ Agent 使用說明

這個工具的用途：把 QAT（量化感知訓練）訓練好的 YOLOv7 模型（`.pt`）自動轉成硬體可用的量化 ONNX（`.onnx`）。

```
best.pt + model.txt  →  [QDQ Agent]  →  model_bias_topo.onnx
```

---

## 你需要準備什麼

| 檔案 | 說明 | 從哪裡來 |
|------|------|----------|
| `best.pt` | QAT 訓練好的模型 | 訓練完後存在 `runs/train/expXX/weights/best.pt` |
| `model.txt` | 模型架構與 FL 值 | 訓練完後從 `runs/train/expXX/opt_after-best.txt` 複製出來 |

> **重要**：`model.txt` 必須跟 `best.pt` 是同一次訓練的結果，不能混用。

---

## 環境設定（第一次用才需要）

**1. 把專案複製到 server**
```bash
git clone https://github.com/jackwang220/QDQ-agent.git
cd QDQ-agent
```

**2. 安裝套件**
```bash
# QDQ pipeline 的套件（需要 GPU server）
pip install -r requirements_qdq.txt

# Agent 的套件
pip install -r requirements_agent.txt
```

**3. 設定 API key**

複製 `.env.template` 成 `.env` 然後填入 key：
```bash
cp .env.template .env
```
打開 `.env` 填入：
```
ANTHROPIC_API_KEY=你的金鑰
```

---

## 每次換模型只需要改這裡

打開 `pipeline_config.yaml`，修改以下兩個地方：

```yaml
model:
  weights: "./best.pt"      # ← 改成你的 .pt 路徑
  img_size: 640             # ← 改成你的模型輸入大小（通常不用改）

paths:
  yolov7_dir: "/home1/m314832018/yolov7"  # ← yolov7 repo 路徑（通常不用改）
  model_txt: "./model.txt"                 # ← 改成你的 model.txt 路徑
                                           #    如果沒有 model.txt 就填 "auto"
```

**其他欄位（`postprocess` 區塊）不用動**，agent 會自動推斷。

---

## 跑 Pipeline

```bash
cd ~/QDQ-agent
python -m qdq_agent.main --config pipeline_config.yaml
```

跑的過程中會有幾個地方需要你按 Enter 確認：

**確認 detect layer**（直接 Enter 接受建議即可）
```
[HUMAN IN THE LOOP] (detect_layer_review)
Agent suggests detect_layer = 105.
Press Enter to accept, or type a different number:
>                          ← 直接按 Enter
```

**確認缺少 Q/DQ 的節點**（直接 Enter 接受全部）
```
[HUMAN IN THE LOOP] (qdq_coverage_review)
Agent found nodes missing Q/DQ coverage.
Press Enter to accept all, or type 'skip' to skip all.
>                          ← 直接按 Enter
```

跑完大約 15-30 秒，完成後輸出在：
```
wrap_all_model_onnx_export/model_bias_topo.onnx   ← 這就是最終結果
```

---

## 沒有 model.txt 怎麼辦

把 `pipeline_config.yaml` 裡的 `model_txt` 設成 `"auto"`：

```yaml
paths:
  model_txt: "auto"
```

Agent 會自動從 `.pt` 生成，但要注意：

- 自動生成的 FL 值**可能不準確**（因為 EMA model 版本問題）
- 建議跑完後和原始 FL 值比對一下
- 如果有差異，建議去找當時訓練的 `opt_after-best.txt` 手動複製

---

## 如果以後要自己訓練

訓練用的 script 是 `code/train_quant_v7.py`（需要在 yolov7 repo 底下跑）。

訓練完成後，在 `runs/train/expXX/` 底下會有：
- `weights/best.pt` → 這就是你要用的模型
- `opt_after-best.txt` → 打開這個檔案，把裡面的 `str(model)` 內容複製出來存成 `model.txt`

建議把 `best.pt` 和 `model.txt` 放在同一個資料夾一起保管。

---

## 常見問題

**Q：Pipeline 跑失敗，說找不到 yolov7 相關的 class**
A：確認 `pipeline_config.yaml` 裡的 `yolov7_dir` 指向正確的 yolov7 repo 路徑。

**Q：自動生成的 model.txt 和交接的 FL 值不一樣**
A：參考「沒有 model.txt 怎麼辦」那一節，用手動複製的方式比較可靠。

**Q：想重新跑一次（清掉上次結果）**
A：刪掉 `wrap_all_model_onnx_export/` 資料夾和 `model.txt`（如果是自動生成的），再重新跑。

**Q：想指定不同的 thread id（避免和上次衝突）**
```bash
python -m qdq_agent.main --config pipeline_config.yaml --thread-id my_run_02
```

---

## 檔案結構說明

```
QDQ-agent/
├── pipeline_config.yaml    ← 唯一需要改的設定檔
├── best.pt                 ← 你的模型（自己放進來）
├── model.txt               ← 訓練時的模型架構（自己放進來）
├── code/                   ← 各步驟的 Python script（通常不需要動）
│   ├── export_quant_fused.py      Step 2
│   ├── model_fp32_int8.py         Step 3
│   ├── onnx_view_wrap_all_repconv_topo.py  Step 4
│   ├── implicit_topo.py           Step 5
│   └── modify_model_topo.py       Step 6
├── qdq_agent/              ← Agent 本體（通常不需要動）
└── wrap_all_model_onnx_export/    ← 輸出資料夾（自動產生）
    └── model_bias_topo.onnx       ← 最終結果
```
