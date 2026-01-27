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

# Port megnyitása (Render automatikusan beállítja a PORT env változót, de dokumentáljuk)
EXPOSE 10000

# Indítás Gunicorn szerverrel
# - bind: Dinamikusan a PORT környezeti változóra (default 10000)
# - workers: 1 (Memória spórolás miatt a Free tier-en)
# - threads: 8 (Hogy több kérést tudjon kezelni egyszerre)
# - timeout: 120 (Hogy a lazy loading ML tanítás ne timeoljon ki)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 8 --timeout 120 run:app"]
