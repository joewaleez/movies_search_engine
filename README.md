# Professional Movie Search Engine

This is a cleaned and professional version of the Data Mining / IR movie search project.

## What was improved

- Removed the huge full inverted-index print because it is not useful in the final project.
- Replaced console input with a professional Streamlit UI.
- Separated backend logic from frontend UI.
- Used TF-IDF + cosine similarity ranking.
- Added optional query expansion using a simple synonym map.
- Added dataset upload, result cards, document source badges, scores, and project statistics.

## Project Structure

```text
professional_movie_search_engine/
│
├── app.py
├── requirements.txt
├── backend/
│   ├── __init__.py
│   └── search_engine.py
└── data/
```

## How to Run

1. Open the folder in VS Code.
2. Put the datasets anywhere on your computer.
3. Install libraries:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

5. Upload:
   - `tmdb_5000_movies.csv`
   - `cleanedData.csv`

Then search normally from the web UI.

## Suggested Test Queries

- action movie
- romantic comedy
- horror film
- science fiction
- animated adventure
