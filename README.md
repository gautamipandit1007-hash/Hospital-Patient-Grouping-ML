# Hospital Patient Grouping using Unsupervised Machine Learning

Groups hospital encounters with similar characteristics and resource use (K-Means, Hierarchical clustering, PCA) and
offers a Streamlit prototype. **Analytical/educational project - clusters are not diagnoses or risk levels.**

Dataset: *Diabetes 130-US Hospitals for Years 1999-2008*, UCI Machine Learning Repository, CC BY 4.0
(https://archive.ics.uci.edu/dataset/296). Citation: Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014).
DOI 10.24432/C5230J.

## 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data
Download from the UCI page above and put **`diabetic_data.csv`** (and `IDS_mapping.csv`) in `data/raw/`.
(`src/data_loader.py` also has an untested `download_dataset()` helper; if it fails, download manually.)

## 3. Run
```bash
python -m src.pipeline            # cleaning, EDA figures, models, metrics -> models/, reports/, data/processed/
python -m src.sample_predictions  # optional: writes docs/sample_outputs.md from docs/sample_inputs.json
streamlit run app/app.py          # the prototype
```
Or explore step by step in `notebooks/hospital_patient_grouping.ipynb` (Jupyter or Colab).
Regenerate the notebook with `python notebooks/build_notebook.py`.

## 4. Folder structure
```
data/raw          original CSV (not committed)       data/processed  cleaned + clustered data, profile tables
src/              config, loader, preprocessing, eda, clustering, evaluation, profiling, pipeline, predict
app/app.py        Streamlit app (7 pages)            notebooks/      step-by-step notebook
models/           saved preprocessor, K-Means, PCA   reports/        figures/ + results.json (all real numbers)
docs/             report template, viva Q&A, architecture, sample inputs/outputs     tests/  smoke test
```

## 5. Reproducibility
- `RANDOM_STATE = 42` everywhere; hierarchical clustering uses a fixed random sample (`HIER_SAMPLE_SIZE`).
- All parameters live in `src/config.py` (k range, minimum k, optional `K_OVERRIDE`, features).
- Every metric shown in the app/report is read from `reports/results.json`, written by actually running the pipeline.

## 6. Design decisions (explain these in your viva)
- One encounter per patient (rows must be independent); expired/hospice discharges removed.
- `readmitted` (outcome), `race`, `gender` are **not** model inputs; they only describe clusters afterwards.
- Outliers are capped at the 99th percentile (not deleted); numeric features standardised; categories one-hot encoded.
- Choosing k: highest silhouette among k >= 3 (k=2 is usually too coarse to profile). Change in `config.py` if justified.
- Hierarchical clustering runs on a 5,000-row sample (O(n^2)); all methods are compared on that same sample and space.
- The app assigns new patients to the nearest **K-Means** centroid because hierarchical clustering has no predict step.

## 7. Testing
`pytest -q` runs the whole pipeline on a **synthetic** file in a temporary copy (your outputs are untouched).
The synthetic data is only for testing code, never for results.

## 8. Mapping to the evaluation criteria
| Criterion (marks) | Where it is covered |
|---|---|
| Problem definition (10) | Report sections 2-6, notebook step 1, app Home |
| Dataset selection and preprocessing (15) | `src/preprocessing.py`, notebook steps 2-4, `results.json -> cleaning` |
| ML algorithms (20) | `src/clustering.py`, notebook steps 5A-5C |
| Comparison and evaluation (20) | `src/evaluation.py`, notebook step 6, app Model Comparison |
| Novelty, interpretation (10) | `src/profiling.py` (data-derived cluster descriptions, utilisation index), leakage/bias handling |
| Prototype (15) | `app/app.py` |
| Report, presentation, viva (10) | `docs/REPORT_TEMPLATE.md`, `docs/VIVA_QA.md` |

## 9. Ethics
De-identified public data; no attempt to re-identify; sensitive attributes excluded from model inputs; clusters are
descriptive similarity groups. See report section 20.
