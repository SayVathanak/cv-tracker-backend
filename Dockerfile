# 1. Start with a lightweight Python system
FROM python:3.10-slim

# 2. Install the "System Tools" (Tesseract for OCR, Poppler for PDF)
# This is the step that makes your app work on the cloud!
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 3. Set up the working folder
WORKDIR /app

# 4. Copy your files into the container
COPY . .

# 5. Install your Python libraries (FastAPI, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Create the folder for uploads
RUN mkdir -p static_uploads

# 7. Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]