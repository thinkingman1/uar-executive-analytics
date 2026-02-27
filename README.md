# UAR Executive Analytics

Streamlit dashboard for UAR analytics: KPIs, reassignment behavior, outcome trends, compliance-map animations, and Sankey path analysis.

## Local Run

```bash
cd /Users/sandeeppadam/Documents/cursor/uar
source .venv/bin/activate
streamlit run app.py
```

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

## Deploy (Streamlit Community Cloud)

1. Push this project to GitHub.
2. In Streamlit Community Cloud, click **New app**.
3. Select your GitHub repo + branch.
4. Set main file path to `app.py`.
5. Deploy.

## Project Structure

- `app.py` - Main Streamlit app
- `modules/` - Supporting chart/profiling utilities
- `requirements.txt` - Python dependencies
