# QDQ Agent — CPU-only, self-contained image for air-gapped deployment.
# 依賴用 uv 從 requirements.lock 完整鎖定版本（可重現）。
# yolov7 (Step 0/2) 的執行期依賴 (opencv/scipy/seaborn/matplotlib…) 一併裝入；
# yolov7 本體仍以 volume 掛入 /app/yolov7。
#
# build 需在有網路的機器；做完 build_offline.sh 產生 tar 搬到 air-gapped server。

FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: torch / onnxruntime 的 OpenMP runtime
# libglib2.0-0: opencv-python-headless 在 slim 需要
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 用 uv 安裝完整鎖定的依賴（含 CPU torch 與 yolov7 執行期依賴）
RUN pip install --no-cache-dir uv
COPY requirements.lock .
RUN uv pip install --system --no-cache -r requirements.lock \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match

# 程式碼（best.pt / model.txt / pipeline_config.yaml / yolov7 用 volume 掛）
COPY qdq_agent ./qdq_agent
COPY code ./code

EXPOSE 8000

CMD ["python", "-m", "qdq_agent.server", "--host", "0.0.0.0", "--port", "8000"]
