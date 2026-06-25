#!/usr/bin/env bash
# 在「有網路」的機器上執行，產生一個自包含的 image tar，
# 搬到 air-gapped server 後只需 docker load + docker compose up，全程不連外網。
#
#   ./build_offline.sh
#
# 產出: qdq-agent-offline.tar.gz

set -euo pipefail

IMAGE="qdq-agent:offline"
OUT="qdq-agent-offline.tar.gz"

echo "==> [1/3] Build image (此步驟會連網抓 torch/onnx 等並烤進 image)"
docker build -t "$IMAGE" .

echo "==> [2/3] Save image 成 tar"
docker save "$IMAGE" | gzip > "$OUT"

echo "==> [3/3] 完成"
ls -lh "$OUT"
echo
echo "把以下檔案一起搬到 air-gapped server："
echo "  - $OUT"
echo "  - docker-compose.yml"
echo "  - .env                 (記得 langfuse 三個變數留空)"
echo "  - pipeline_config.yaml (yolov7_dir 設成 ./yolov7)"
echo "  - best.pt / model.txt"
echo "  - yolov7/  (整個 repo)"
echo
echo "server 上執行："
echo "  docker load -i $OUT"
echo "  docker compose up -d"
echo "  # 瀏覽器開 http://<server-ip>:8000"
