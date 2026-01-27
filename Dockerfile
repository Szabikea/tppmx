# Hivatalos Playwright Python image használata (tartalmazza a böngészőket és függőségeket)
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

# Munkakönyvtár
WORKDIR /app

# Python függőségek telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Alkalmazás kód másolása
COPY . .

# Környezeti változók
ENV FLASK_APP=run.py
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Port megnyitása
EXPOSE 10000

# Indítás Gunicorn szerverrel
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "--threads", "4", "--timeout", "120", "run:app"]
