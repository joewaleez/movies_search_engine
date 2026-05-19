import streamlit as st

from backend.search_engine import MovieSearchEngine


st.set_page_config(
    page_title="Movie Search Engine",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0px;
}
.subtitle {
    color: #777;
    font-size: 18px;
    margin-top: 0px;
}
.result-card {
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(150,150,150,0.25);
    margin-bottom: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(120,120,120,0.15);
    font-size: 13px;
    margin-right: 8px;
}
.score {
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_engine(tmdb_file, wiki_file):
    engine = MovieSearchEngine()
    engine.load_data(tmdb_file, wiki_file)
    engine.build_index()
    return engine


st.markdown('<p class="main-title">🎬 Professional Movie Search Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">TF-IDF retrieval system with ranking, query expansion, and a clean frontend/backend structure.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Upload Datasets")
    tmdb_file = st.file_uploader("Upload TMDB CSV", type=["csv"])
    wiki_file = st.file_uploader("Upload Wikipedia Movies CSV", type=["csv"])

    st.divider()
    st.header("⚙️ Search Settings")
    top_k = st.slider("Number of results", 5, 30, 10)
    use_expansion = st.toggle("Use query expansion", value=True)

    st.info("Required columns: TMDB uses title/overview. Wikipedia can use title, genre, category, Plot if available.")

if tmdb_file is None or wiki_file is None:
    st.warning("Upload the two datasets from the sidebar to start searching.")
    st.stop()

try:
    with st.spinner("Building the search index..."):
        engine = get_engine(tmdb_file, wiki_file)
except Exception as e:
    st.error(f"Could not load/build the search engine: {e}")
    st.stop()

stats = engine.stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Documents", f"{stats['documents']:,}")
c2.metric("TMDB Docs", f"{stats['tmdb']:,}")
c3.metric("Wikipedia Docs", f"{stats['wikipedia']:,}")
c4.metric("TF-IDF Features", f"{stats['features']:,}")

st.divider()

query = st.text_input(
    "Search for a movie topic",
    placeholder="Examples: action movie, romantic comedy, horror film, science fiction..."
)

if query:
    results = engine.search(query, top_k=top_k, use_expansion=use_expansion)

    st.subheader(f"Search Results for: {query}")

    if not results:
        st.error("No relevant documents found.")
    else:
        for item in results:
            st.markdown(
                f"""
                <div class="result-card">
                    <h3>#{item.rank} — {item.title}</h3>
                    <span class="badge">{item.source}</span>
                    <span class="badge">Doc ID: {item.doc_id}</span>
                    <span class="badge score">Score: {item.score}</span>
                    <p>{item.description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("Write a query above to test the search engine.")
