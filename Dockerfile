# QDQ Agent — CPU-only, self-contained image for air-gapped deployment.
# 依賴用 uv 從 requirements.lock 完整鎖定版本（可重現）。
# torch/torchvision 從 PyTorch CPU index 單獨安裝，其餘走 PyPI（避免對整個 lock
# 查詢 pytorch index 造成極慢）。yolov7 (Step 0/2) 執行期依賴一併裝入；
# yolov7 本體仍以 volume 掛入 /app/yolov7。

FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: torch / onnxruntime 的 OpenMP；libglib2.0-0: opencv-python-headless 需要
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir uv

# (1) torch / torchvision — 只從 PyTorch CPU index 抓（2.0.1 < 2.6，torch.load 安全）
RUN uv pip install --system --no-cache \
      torch==2.0.1+cpu torchvision==0.15.2+cpu \
      --index-url https://download.pytorch.org/whl/cpu

# (2) 其餘鎖定依賴 — 走 PyPI（含 QDQ agent + onnx + yolov7 執行期 opencv/scipy/seaborn…）
COPY requirements.lock .
RUN uv pip install --system --no-cache -r requirements.lock

COPY qdq_agent ./qdq_agent
COPY code ./code

EXPOSE 8000

CMD ["python", "-m", "qdq_agent.server", "--host", "0.0.0.0", "--port", "8000"]
