FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY setup.py .          # ✅ copy before pip install
COPY README.md .         # ✅ needed too since setup.py reads it

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]