FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8501

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render/Hugging Face/any container host: listens on $PORT
CMD streamlit run app/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
