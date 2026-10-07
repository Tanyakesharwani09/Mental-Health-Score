# Mansik Santulan Score — Student Mental Health Predictor

**Author:** Tanya Kesharwani

An end-to-end machine learning project that predicts a student's mental health
score (0–10) from their social media usage patterns and lifestyle habits —
screen time, phone unlocks, study hours, physical activity, sleep, and
perceived stress level.

## Project structure

| File | Purpose |
|---|---|
| `ML_Project.ipynb` | Full ML workflow: EDA, cleaning, feature engineering, model comparison, tuning |
| `Student Social Media And Mental Health Impact.csv` | Dataset — 5,000 student records |
| `Mental_Health_Model.pkl` | Saved scikit-learn pipeline (preprocessing + Random Forest) |
| `main.py` | FastAPI backend with Pydantic validation, `/predict` endpoint |
| `index.html` / `style.css` / `script.js` | Frontend — form, validation, animated score gauge |

## Tech stack

- **ML:** Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib
- **API:** FastAPI, Pydantic, Uvicorn
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Deployment:** Render

## ML approach

- Cleaned 5,000-row survey dataset (deduped, clipped impossible values)
- Feature engineering: grouped 111 countries into top-10 + "Other" to control cardinality
- `ColumnTransformer` preprocessing: `log1p` + scaling for skewed features,
  `OrdinalEncoder` for stress level (ordered Low → Very High),
  `OneHotEncoder` for nominal columns
- Compared Linear Regression baseline vs. Random Forest; tuned the forest with `RandomizedSearchCV`
- Exported the full pipeline so inference uses identical preprocessing

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --port 2200 --reload   # backend
python -m http.server 8000              # frontend → http://localhost:8000
```

Point `API_BASE` in `script.js` at your backend URL if it differs.

*Disclaimer: informational tool only — not a clinical assessment.*
