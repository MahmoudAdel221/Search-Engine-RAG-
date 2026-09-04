
import os
import re
import sys
import uuid
import pickle

import nltk
import numpy as np
import fitz                  # PyMuPDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import chromadb

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── NLTK downloads ────────────────────────────────────────────────────────────
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

# ── Globals ───────────────────────────────────────────────────────────────────
english_stopwords = set(stopwords.words('english'))
lemmatizer        = WordNetLemmatizer()

CHROMA_PATH      = "./chroma_store"          # persistent directory for ChromaDB
COLLECTION_NAME  = "pdf_tfidf"
VECTORIZER_PATH  = "./tfidf_vectorizer.pkl"  # saved fitted TF-IDF vectorizer
CHUNK_SIZE       = 500                       # characters per chunk
CHUNK_OVERLAP    = 50                        # overlap between consecutive chunks


def preprocess_text(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    cleaned = []
    for tok in tokens:
        if not tok.isalpha():
            continue
        if tok in english_stopwords:
            continue
        cleaned.append(lemmatizer.lemmatize(tok, pos='v'))
    return cleaned


def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\b\d+/\d+\b', '', text)
    text = re.sub(r'[•\uf0b7\uf0a7◦▪▸●○□■–—]', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([a-z])(E\.g\.|I\.e\.)', r'\1 \2', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from a PDF using fitz (PyMuPDF) native text extraction.
    Returns a list of (page_number, cleaned_text) tuples."""
    doc   = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append((i, clean_text(text)))

    doc.close()
    return pages


def chunk_text_with_pages(
    pages:      list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap:    int = CHUNK_OVERLAP,
) -> list[tuple[str, int]]:
    """Split each page's text into overlapping chunks.
    Returns list of (chunk_text, page_number) tuples."""
    results = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end   = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                results.append((chunk, page_num))
            start += chunk_size - overlap
    return results


def build_tfidf_vectorizer(texts: list[str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_text,
        preprocessor=None,
        lowercase=False,
        min_df=2,
        max_df=0.9,
    )
    vectorizer.fit(texts)
    return vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path: str = VECTORIZER_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(path: str = VECTORIZER_PATH) -> TfidfVectorizer:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_chroma_collection(persist_dir: str = CHROMA_PATH):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name               = COLLECTION_NAME,
        metadata           = {"hnsw:space": "cosine"},
        embedding_function = None,
    )
    return client, collection


def pdf_already_indexed(collection, pdf_name: str) -> bool:
    results = collection.get(where={"source": pdf_name}, limit=1)
    return len(results["ids"]) > 0


def index_chunks(collection,chunks:     list[str],vectors:    np.ndarray,pdf_name:   str,pages:      list[int]) -> None:          # ← real page numbers
    
    ids       = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "source":      pdf_name,
            "page_number": pages[i],                      # ← real PDF page
            "text":        chunks[i][:500],
        }
        for i in range(len(chunks))
    ]
    collection.add(
        ids        = ids,
        embeddings = vectors.tolist(),
        documents  = chunks,
        metadatas  = metadatas,
    )


def build_index(pdf_files: list[str], collection) -> TfidfVectorizer:
    new_pdf_files = [p for p in pdf_files
                     if not pdf_already_indexed(collection, os.path.basename(p))]

    if not new_pdf_files:
        print("All PDFs are already indexed.")
        if os.path.exists(VECTORIZER_PATH):
            print("Loading existing TF-IDF vectorizer ...")
            return load_vectorizer()
        else:
            raise FileNotFoundError(
                "Vectorizer file not found. Delete the chroma_store folder "
                "and re-run to rebuild from scratch."
            )

    all_chunks    = []
    all_pages     = []                                     # ← page number per chunk
    chunk_sources = []

    for pdf_path in new_pdf_files:
        pdf_name = os.path.basename(pdf_path)
        print(f"  [extract] '{pdf_name}' ...")
        pages = extract_text_from_pdf(pdf_path)            # [(page_num, text), ...]
        if not pages:
            print(f"  [warn] No extractable text in '{pdf_name}'. Skipping.")
            continue
        chunks_with_pages = chunk_text_with_pages(pages)   # [(chunk, page_num), ...]
        print(f"           → {len(chunks_with_pages)} chunks across {len(pages)} page(s)")
        for chunk, page_num in chunks_with_pages:
            all_chunks.append(chunk)
            all_pages.append(page_num)
            chunk_sources.append(pdf_name)

    if not all_chunks:
        raise ValueError("No text could be extracted from any PDF.")

    print(f"\n  [tfidf] Fitting TF-IDF on {len(all_chunks)} chunks ...")
    vectorizer = build_tfidf_vectorizer(all_chunks)
    tfidf_matrix = vectorizer.transform(all_chunks)
    save_vectorizer(vectorizer)
    print(f"  [tfidf] Vocabulary size: {len(vectorizer.vocabulary_)} terms")

    for pdf_path in new_pdf_files:
        pdf_name = os.path.basename(pdf_path)
        indices = [i for i, s in enumerate(chunk_sources) if s == pdf_name]
        if not indices:
            continue
        start, end  = indices[0], indices[-1] + 1
        pdf_chunks  = all_chunks[start:end]
        pdf_vectors = tfidf_matrix[start:end].toarray()
        pdf_pages   = all_pages[start:end]                 # ← page numbers slice
        print(f"  [store] '{pdf_name}' → {len(pdf_chunks)} vectors into ChromaDB ...")
        index_chunks(collection, pdf_chunks, pdf_vectors, pdf_name, pdf_pages)

    print("  [done] Index built.\n")
    return vectorizer


def search(query: str, vectorizer: TfidfVectorizer, collection, top_k: int = 5) -> None:
    query = query.strip()
    if not query:
        print("Empty query, please enter some text.")
        return

    query_vec = vectorizer.transform([query])               # (1, vocab_size) sparse

    # ── Retrieve ALL stored embeddings from ChromaDB ──────────────────────────
    stored = collection.get(
        include=["embeddings", "documents", "metadatas"],
    )

    if not stored["ids"]:
        print("No results found.")
        return

    stored_embeddings = np.array(stored["embeddings"])       # (n_chunks, vocab_size)

    # ── Compute cosine similarity using scikit-learn ──────────────────────────
    similarities = cosine_similarity(query_vec, stored_embeddings)[0]   # (n_chunks,)

    # ── Rank by descending similarity and pick top_k ──────────────────────────
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print("====================================")
    print(f"\nQuery: {query}")
    print("====================================")

    for rank, idx in enumerate(top_indices, start=1):
        sim      = similarities[idx]
        doc      = stored["documents"][idx]
        meta     = stored["metadatas"][idx]
        source   = meta.get("source",      "unknown")
        page_num = meta.get("page_number", "?")             # ← real page number
        snippet  = doc

        print(f"""
┌─ Result #{rank} {'─'*40}
│  File       : {source}
│  Page       : {page_num}
│  Similarity : {sim:.1%}
│
│  {snippet}
└{'─'*50}""")


def main() -> None:
    pdf_folder = r"pdfs"
    pdf_files  = [
        os.path.join(pdf_folder, f)
        for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in '{pdf_folder}'.")
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDF(s) in '{pdf_folder}'.")
    print(f"Connecting to ChromaDB at '{CHROMA_PATH}' ...")
    _, collection = get_chroma_collection()

    print("\nBuilding TF-IDF index (this may take a moment) ...")
    vectorizer = build_index(pdf_files, collection)

    total_docs = collection.count()
    print(f"Index ready – {total_docs} total chunk(s) in the collection.\n")

    print("Search engine - Type your query.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Query > ")
        if query.lower() in {"exit", "quit"}:
            print("Goodbye")
            break
        search(query, vectorizer, collection, top_k=5)


if __name__ == "__main__":
    main()
