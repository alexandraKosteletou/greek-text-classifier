# 🇬🇷 Greek Text Classifier (TF‑IDF + Linear SVM)
![CI](https://github.com/alexandraKosteletou/greek-text-classifier/actions/workflows/ci.yml/badge.svg)

Πρακτικό NLP project για ταξινόμηση ελληνικών (και μη) κειμένων με **TF‑IDF + Linear SVM**.
Περιλαμβάνει CLI, FastAPI service, tests, GitHub Actions CI, και φάκελο με *legacy demos* (NLTK/Scikit‑Learn scripts).

## Περιεχόμενα
- Εκπαίδευση και αποθήκευση μοντέλου (`artifacts/model.joblib`)
- CLI: `train`, `predict`
- FastAPI API: `/predict`, `/health`
- Pytest tests + CI (κατεβάζει αυτόματα NLTK corpora)
- Legacy demos (τρέχουν στο CI για smoke)

## Γρήγορα
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m spacy download xx_sent_ud_sm || true  # προαιρετικό για απλή tokenization
```

### Εκπαίδευση
```bash
python -m nlp_project.train --data data/sample.csv --model artifacts/model.joblib
```

### Πρόβλεψη (CLI)
```bash
python -m nlp_project.predict --model artifacts/model.joblib --text "Αυτό είναι υπέροχο!"
```

### API
```bash
uvicorn nlp_project.api:app --reload
# POST /predict  { "text": "Το προϊόν ήταν χάλια" }
```

## Legacy demos (kept as‑is)
Μικρά scripts/demos, φιλτραρισμένα ώστε να μην υπάρχουν διπλές/προβληματικές εκδόσεις.
Τρέχουν ανεξάρτητα από το κύριο pipeline. Δες: `src/nlp_project/legacy/`

- `bag_of_words.py`
- `category_predictor.py`
- `frequency.py`
- `frequency_distribution.py`
- `in_built_corpora.py`
- `lemmatizer_demo.py`
- `lookup_tagger.py`
- `play_training_set.py`
- `sentiment_naive_bayes.py` *(χρησιμοποιεί `data/movies.csv`)*
- `sentiment_vader.py`
- `sklearn_exercise.py`
- `stemmer_demo.py`
- `stemmer_greek_demo.py` *(προαιρετικό: απαιτεί `greek_stemmer`)*
- `test_bigram_tagger.py`
- `test_tagger_mystery_model.py`
- `test_unigram_tagger_split.py`
- `tokenizer_demo.py`
- `text_chunker.py`

### Χρήση legacy hooks στο pipeline
Αν υπάρχει `src/nlp_project/legacy/custom_hooks.py`, το pipeline θα χρησιμοποιήσει **custom hooks**:

- `custom_preprocess(text)` → lowercase + URL masking + (προαιρετικό) Greek stemming ή English lemmatization
- `custom_tokenize(text)` → NLTK `word_tokenize` όπου διαθέσιμο, αλλιώς regex split
- `custom_vectorizer()` → από προεπιλογή `TfidfVectorizer`, με env vars:
  - `LEGACY_VEC=count|tfidf`
  - `LEGACY_NGRAMS=1,2`
  - `LEGACY_MAX_FEATURES=50000`

Παράδειγμα:
```bash
export LEGACY_VEC=count LEGACY_NGRAMS=1,3 LEGACY_MAX_FEATURES=100000
python -m nlp_project.train --data data/sample.csv --model artifacts/model.joblib
```

> Σημείωση: Το `greek_stemmer` είναι **προαιρετικό**. Αν το χρειάζεσαι για ελληνικό stemming:
> ```bash
> pip install greek-stemmer==0.1.1
> ```

## Δομή
```
.
├─ src/nlp_project/
│  ├─ data.py, features.py, model.py, train.py, predict.py, api.py
│  ├─ legacy/
│  │   ├─ custom_hooks.py
│  │   └─ (demos *.py)
├─ data/
│  ├─ sample.csv
│  └─ movies.csv (προαιρετικό για demo)
├─ tests/
│  └─ test_api.py
├─ scripts/
│  └─ download_nltk.py
├─ .github/workflows/ci.yml
├─ requirements.txt
├─ pyproject.toml
└─ Makefile, Dockerfile (αν προστεθούν)
```

## CI
Το GitHub Actions workflow:
- Εγκαθιστά εξαρτήσεις
- Κατεβάζει NLTK corpora/models (`scripts/download_nltk.py`)
- Τρέχει `pytest`
- Τρέχει *όλα* τα legacy demos (smoke)

---

**License**: MIT (ή ό,τι επιλέξεις)
