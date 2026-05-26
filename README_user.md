# QDQ Agent 使用說明

把 QAT 訓練好的模型（`.pt`）自動轉成硬體可部署的量化 ONNX。

```
best.pt + model.txt  →  [QDQ Agent]  →  model_bias_topo.onnx
```

---

## 準備檔案

| 檔案 | 從哪裡來 |
|------|----------|
| `best.pt` | 訓練結果 `runs/train/expXX/weights/best.pt` |
| `model.txt` | 訓練結果 `runs/train/expXX/opt_after-best.txt`，直接複製改名 |

> `model.txt` 必須跟 `best.pt` 來自**同一次訓練**，不能混用。

---

## 每次換模型只需改這裡

打開 `pipeline_config.yaml`，改這兩個地方：

```yaml
model:
  weights: "./best.pt"     # 你的 .pt 路徑
  img_size: 640            # 模型輸入大小（通常不用改）

paths:
  model_txt: "./model.txt" # 你的 model.txt 路徑；沒有的話填 "auto"
```

其他欄位不用動，agent 自動推斷。

---

## 啟動方式

### 方式 A — Web Panel（推薦）

SSH 進 server 後執行：

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ivs && cd ~/QDQ-agent && python -m qdq_agent.server --host 0.0.0.0 --port 8000
```

瀏覽器開 `http://<server-ip>:8000`，在左側填好設定後按 **Start Pipeline**。

Pipeline 跑的過程中會在瀏覽器跳出幾個確認步驟（HITL），直接按「Accept」即可。

### 方式 B — 終端機 CLI

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ivs && cd ~/QDQ-agent
python -m qdq_agent.main --config pipeline_config.yaml
```

遇到 `[HUMAN IN THE LOOP]` 直接按 Enter 接受建議即可。

---

## 輸出

```
wrap_all_model_onnx_export/model_bias_topo.onnx   ← 最終結果
```

全程約 15–30 秒。

---

## 沒有 model.txt

把 `model_txt` 設成 `"auto"`，agent 會從 `.pt` 自動生成。
但自動生成的 FL 值**可能不準確**，建議優先用訓練時存下來的版本。

---

## 重新跑（清除上次結果）

```bash
rm -rf wrap_all_model_onnx_export/
python -m qdq_agent.main --config pipeline_config.yaml --thread-id run_02
```

---

## 關於 yolov7

Pipeline 的 Step 0（生成 model.txt）和 Step 2（匯出 raw ONNX）需要用到實驗室內部的 yolov7 框架。

**這不是 pip 套件**，是公司/實驗室自己 fork 並修改過的版本，需要向訓練模型的學長索取本地 repo。取得後設定路徑：

```yaml
paths:
  yolov7_dir: "/home1/m314832018/yolov7"   # server 上已有，通常不用改
```

**可以跳過 yolov7 的情況：**

| 條件 | 跳過的 Step |
|------|------------|
| 已有 `model.txt` | Step 0 自動跳過 |
| 已有 `best_sim_annotate.onnx`（放在 `.pt` 旁邊） | Step 2 自動跳過 |

若使用的 server 已備好 `yolov7` repo 和 `best_sim_annotate.onnx`，正常使用不會碰到這個問題。換到新的 server 時才需要重新準備。

---

## 常見問題

**Q：說找不到 yolov7 相關 class**
A：`yolov7_dir` 要指向實驗室內部的 yolov7 repo，這不是 pip 套件，詳見下方「關於 yolov7」段落。如果已有 `best_sim_annotate.onnx` 放在 weights 旁邊，Step 2 會自動跳過，不需要 yolov7。

**Q：想保留上次結果重跑**
A：用 `--thread-id` 指定新的 id，不會覆蓋上次的 checkpoint。

**Q：SSH 斷線後想讓 server 繼續跑**
A：改用 tmux 啟動：
```bash
tmux new-session -d -s qdq 'conda activate ivs && cd ~/QDQ-agent && python -m qdq_agent.server --host 0.0.0.0 --port 8000'
```
之後用 `tmux attach -t qdq` 查看 log。
