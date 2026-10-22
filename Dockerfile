FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    PORT=8000 \
    PORT_HEALTH=8000 \
    CODESCALER_MODEL=LARK-Lab/CodeScaler-4B

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG CODESCALER_REVISION=main
ENV CODESCALER_REVISION=${CODESCALER_REVISION}
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['CODESCALER_MODEL'], revision=os.environ['CODESCALER_REVISION'], allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])"
ENV HF_HUB_OFFLINE=1

COPY server.py .
EXPOSE 8000
CMD ["sh", "-c", "exec python -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
