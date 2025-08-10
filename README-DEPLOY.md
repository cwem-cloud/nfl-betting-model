# Deploy Options

Pick **one** of the quick deploys below.

## A) Streamlit Community Cloud (fastest, free)
1. Push this folder to a **public GitHub repo**.
2. Go to share.streamlit.io → **New app** → select your repo/branch.
3. Main file: `app/app.py`. Python: 3.11. Requirements: `requirements.txt` (auto).
4. Click **Deploy**. After it boots, you'll have a public URL.
5. In the app, click **🔄 Refresh** to retrain/score. (For daily refresh, see GitHub Actions below.)

## B) Hugging Face Spaces (free)
1. Create a Space → **Streamlit** template.
2. Upload everything or connect your GitHub repo.
3. Spaces auto-installs `requirements.txt` and exposes the app.

## C) Render.com (one-click Docker deploy)
1. Push to GitHub. In Render, create a **Web Service** → **Use Docker**.
2. Render picks up `Dockerfile`. Default port 8501.
3. Deploy → you get a public URL. Free plan sleeps when idle.

## D) Docker anywhere
```bash
docker build -t nfl-model .
docker run -p 8501:8501 nfl-model
# Open http://localhost:8501
```

## Scheduled Refresh (optional)
- We include `.github/workflows/daily_refresh.yml` which runs `scripts/refresh.py` daily.
- Add your real data pulls inside `scripts/refresh.py` and commit targets (e.g., precomputed `data/processed/picks_latest.csv`).

## Environment Variables (if needed later)
- Put secrets (API keys) in your cloud’s secret manager (Streamlit Cloud → **Secrets**, Render → **Environment**).
- Access in code via `os.environ.get("YOUR_KEY")`.
