import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchResult:
    rank: int
    doc_id: int
    title: str
    source: str
    description: str
    score: float


class MovieSearchEngine:
    """
    Clean backend for the Movie Search Engine.

    Features:
    - Loads TMDB + Wikipedia movie datasets.
    - Cleans missing values safely.
    - Builds a TF-IDF vector space model.
    - Searches using cosine similarity.
    - Optional lightweight query expansion using a built-in synonym map.
    """

    def __init__(self, max_features: int = 50000):
        self.max_features = max_features
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix = None
        self.synonyms: Dict[str, List[str]] = {
            "action": ["fight", "battle", "war", "hero"],
            "romance": ["love", "romantic", "relationship"],
            "comedy": ["funny", "humor", "comic"],
            "horror": ["scary", "fear", "ghost", "terror"],
            "science": ["sci", "fiction", "space", "future"],
            "fiction": ["sci", "science", "future", "space"],
            "crime": ["detective", "police", "murder"],
            "animated": ["animation", "cartoon"],
            "adventure": ["journey", "quest"],
        }

    @staticmethod
    def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
        if name in df.columns:
            return df[name].fillna("").astype(str)
        return pd.Series([""] * len(df), index=df.index)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_data(self, tmdb_file, wiki_file) -> pd.DataFrame:
        tmdb = pd.read_csv(tmdb_file)
        wiki = pd.read_csv(wiki_file, on_bad_lines="skip", quotechar='"', encoding="utf-8")

        tmdb_part = pd.DataFrame({
            "title": self._safe_col(tmdb, "title"),
            "description": self._safe_col(tmdb, "overview"),
            "genre": self._safe_col(tmdb, "genres"),
            "category": "",
            "source": "TMDB",
        })

        wiki_part = pd.DataFrame({
            "title": self._safe_col(wiki, "title").where(self._safe_col(wiki, "title") != "", self._safe_col(wiki, "Title")),
            "description": self._safe_col(wiki, "Plot"),
            "genre": self._safe_col(wiki, "genre").where(self._safe_col(wiki, "genre") != "", self._safe_col(wiki, "Genre")),
            "category": self._safe_col(wiki, "category").where(self._safe_col(wiki, "category") != "", self._safe_col(wiki, "Category")),
            "source": "Wikipedia",
        })

        df = pd.concat([tmdb_part, wiki_part], ignore_index=True)
        df["title"] = df["title"].fillna("").astype(str)
        df["description"] = df["description"].fillna("").astype(str)
        df["genre"] = df["genre"].fillna("").astype(str)
        df["category"] = df["category"].fillna("").astype(str)

        df["search_text"] = (
            (df["title"] + " ") * 3
            + df["description"] + " "
            + df["genre"] + " "
            + df["category"]
        ).apply(self._clean_text)

        df = df[df["search_text"].str.len() > 0].reset_index(drop=True)
        self.df = df
        return df

    def build_index(self) -> None:
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded. Call load_data() first.")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.df["search_text"])

    def expand_query(self, query: str) -> str:
        words = self._clean_text(query).split()
        expanded = list(words)

        for word in words:
            expanded.extend(self.synonyms.get(word, []))

        # Remove duplicates while preserving order
        seen = set()
        final_terms = []
        for term in expanded:
            if term not in seen:
                final_terms.append(term)
                seen.add(term)

        return " ".join(final_terms)

    def search(self, query: str, top_k: int = 10, use_expansion: bool = True) -> List[SearchResult]:
        if self.df is None or self.vectorizer is None or self.doc_matrix is None:
            raise ValueError("Index is not built. Load data and build index first.")

        query_text = self.expand_query(query) if use_expansion else self._clean_text(query)

        if not query_text:
            return []

        query_vector = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vector, self.doc_matrix).flatten()

        top_indices = scores.argsort()[::-1][:top_k]
        results: List[SearchResult] = []

        for rank, doc_id in enumerate(top_indices, start=1):
            score = float(scores[doc_id])
            if score <= 0:
                continue

            row = self.df.iloc[doc_id]
            results.append(SearchResult(
                rank=rank,
                doc_id=int(doc_id),
                title=row["title"] if row["title"] else f"Document {doc_id}",
                source=row["source"],
                description=row["description"][:500] if row["description"] else row["search_text"][:500],
                score=round(score, 4),
            ))

        return results

    def stats(self) -> Dict[str, int]:
        if self.df is None:
            return {"documents": 0, "tmdb": 0, "wikipedia": 0, "features": 0}

        return {
            "documents": int(len(self.df)),
            "tmdb": int((self.df["source"] == "TMDB").sum()),
            "wikipedia": int((self.df["source"] == "Wikipedia").sum()),
            "features": int(len(self.vectorizer.get_feature_names_out())) if self.vectorizer else 0,
        }
