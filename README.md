# DeepStrainAPI

Separate deployable FastAPI website for DeepStrain segmentation jobs.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open http://127.0.0.1:10000

## Render

Use `render.yaml` and set `REPLICATE_API_TOKEN` in the Render dashboard.
