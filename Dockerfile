FROM python:3.12-slim

WORKDIR /app

# Sistem bagimliliklar (matplotlib icin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ dizini volume olarak baglanacak
VOLUME ["/app/data"]

EXPOSE 8080

# Varsayilan: Telegram bot
CMD ["python", "main.py"]
