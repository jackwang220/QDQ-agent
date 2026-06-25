# QDQ Agent — Docker 離線部署說明

只包 **QDQ 轉檔 agent**（非 QAT 訓練）。純 CPU，不需 GPU。
符合「image 自包含、部署端不連外網」的規定。

```
best.pt + model.txt + yolov7/  →  [QDQ Agent 容器]  →  model_bias_topo.onnx
```

---

## 一、依賴盤點（兩種外網，都已處理）

| 類型 | 內容 | 處理方式 |
|------|------|----------|
| Build-time 套件 | torch(CPU)/onnx/onnxruntime/fastapi… | build 時抓好烤進 image，部署端不再下載 |
| Run-time LLM | agent 決策節點呼叫 LLM | 已指向**內網** `140.113.228.217:8000`，不出外網 |
| Run-time 追蹤 | Langfuse | ⚠️ 預設連 `cloud.langfuse.com` → `.env` 三個變數**留空關閉** |

> Langfuse 必關，否則 air-gapped 下每次跑都會卡在連雲端逾時。見 `.env.docker.example`。

---

## 二、Build（在「有網路」的機器上做一次）

```bash
./build_offline.sh
```

產出 `qdq-agent-offline.tar.gz`（已含所有 Python 套件與系統依賴）。

---

## 三、搬到 air-gapped server

把這些一起搬過去（同一個資料夾）：

| 檔案 | 說明 |
|------|------|
| `qdq-agent-offline.tar.gz` | image 本體 |
| `docker-compose.yml` | 啟動設定 |
| `.env` | 由 `.env.docker.example` 複製；langfuse 留空 |
| `pipeline_config.yaml` | `yolov7_dir` 設成 `./yolov7` |
| `best.pt` / `model.txt` | 要轉的模型（同一次訓練） |
| `yolov7/` | 整個 yolov7 repo（Step 2 `import models` 用） |

---

## 四、Server 上啟動（全程不連外網）

```bash
docker load -i qdq-agent-offline.tar.gz
docker compose up -d
# 瀏覽器開 http://<server-ip>:8000 → 填設定 → Start Pipeline
```

輸出在 `./wrap_all_model_onnx_export/model_bias_topo.onnx`。

純 CLI（不用 web panel）：
```bash
docker compose run --rm qdq-agent \
  python -m qdq_agent.main --config pipeline_config.yaml
```

---

## 四之一、連不到 web panel？（Linux server 常見坑）

| 症狀 | 原因 / 解法 |
|------|------------|
| `localhost:8000` 連不上 | panel 跑在**遠端 server**，要用 `http://<server-ip>:8000`。查 IP：`hostname -I \| awk '{print $1}'` |
| 容器內報 `Failed to resolve 'host.docker.internal'` | Linux Docker 不解析此名稱。compose 已加 `extra_hosts: host.docker.internal:host-gateway`；或直接用明確 IP / `172.17.0.1`（docker bridge gateway） |
| 想從筆電用 `localhost:8000` 開 | SSH 埠轉發（背景執行）：`ssh -fNL 8000:localhost:8000 <user>@<server-ip>`。關閉：`ps aux \| grep "ssh -fNL"` 找 PID 再 `kill`。VSCode 的 Ports 面板轉發也是做同一件事 |

---

## 五、換模型

直接換掉 `best.pt` / `model.txt`（或改 `docker-compose.yml` 的 volume 來源），
**不用重建 image**，重跑 `docker compose up` 即可。

---

## 六、若安全規定連「build 也必須離線」

上面的做法 build 在有網機器、部署端離線，已符合一般「image 自包含」要求。
若資安要求**連 build 都不可碰網路**（需在 server 上用 vendored 套件重建），
告訴我，我再補一份 `pip download` 預抓 wheels + `--no-index` 安裝的 `Dockerfile.vendored` 變體。
