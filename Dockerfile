FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY setup.py .        
COPY README.md .         

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]