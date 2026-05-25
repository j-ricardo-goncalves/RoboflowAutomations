FROM python:3.11-slim

WORKDIR /a2

# install dependencies first (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . .

ENV PYTHONPATH=/a2

# /tmp/a2 is the default WORKING_DIR — override with env var if needed
# mount a local image folder here to skip fetching from Roboflow:
#   -v /your/images:/tmp/local_images   then set LOCAL_IMAGE_DIR=/tmp/local_images
VOLUME ["/tmp/a2"]

ENTRYPOINT ["python", "main.py"]
