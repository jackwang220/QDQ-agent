# QDQ Agent — CPU-only, self-contained image for air-gapped deployment.
# 只包 QDQ 轉檔 agent（非 QAT 訓練）。所有套件烤進 image，部署端無需連外網。
#
# 注意：build 必須在「有網路」的機器上做（pip/apt 在這裡抓好並烤進 image）。
#       做完用 build_offline.sh 產生 .tar 搬到 air-gapped server 用 docker load。

# 釘死 amd64：這樣即使在 arm64 Mac 上 build，產出的也是 server(x86_64) 能跑的 image
# （參考 rust-translator 的 Dockerfile 同樣這樣釘）
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# onnxruntime / torch 需要 OpenMP runtime；build 時抓，烤進 image
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) 先裝 CPU 版 torch（強制走 PyTorch CPU index，避免抓到上 GB 的 CUDA build）
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# 2) 其餘依賴（torch 已滿足，不會被重裝）
COPY requirements_agent.txt requirements_qdq.txt ./
RUN pip install --no-cache-dir -r requirements_agent.txt -r requirements_qdq.txt

# 3) 程式碼（會變動的 best.pt / model.txt / config / yolov7 用 volume 掛，不烤進來）
COPY qdq_agent ./qdq_agent
COPY code ./code

EXPOSE 8000

# 預設啟動 web panel；要跑純 CLI 可在 docker run 後面覆寫 command
CMD ["python", "-m", "qdq_agent.server", "--host", "0.0.0.0", "--port", "8000"]
