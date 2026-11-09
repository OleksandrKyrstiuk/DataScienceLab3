"""Task 7: Integrated NLP Pipeline."""
import inspect
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.task1_text_cleaning import clean_and_tokenize
from src.task2_stemm_lemma import stem_and_lemmatize
from src.task5_word2vec import train_word2vec
from src.task6_Ngrams import generate_ngrams, count_ngrams


def _apply_stem_or_lemma(tokens: List[str], use_lemmatization: bool) -> List[str]:
    """Helper function to safely invoke stem_and_lemmatize and return List[str]."""
    try:
        sig = inspect.signature(stem_and_lemmatize)
        params = sig.parameters

        if "use_lemmatization" in params:
            res = stem_and_lemmatize(tokens, use_lemmatization=use_lemmatization)
        elif "method" in params:
            method = "lemma" if use_lemmatization else "stem"
            res = stem_and_lemmatize(tokens, method=method)
        elif "lemmatize" in params:
            res = stem_and_lemmatize(tokens, lemmatize=use_lemmatization)
        else:
            try:
                res = stem_and_lemmatize(tokens, use_lemmatization)
            except TypeError:
                res = stem_and_lemmatize(tokens)
    except Exception:
        res = stem_and_lemmatize(tokens)

    if isinstance(res, tuple):
        chosen = res[1] if use_lemmatization else res[0]
        return [str(x) for x in chosen]
    elif isinstance(res, dict):
        if use_lemmatization:
            chosen = res.get('lemmatized') or res.get('lemma') or res.get('lemmas')
        else:
            chosen = res.get('stemmed') or res.get('stem') or res.get('stems')
        if chosen is None:
            chosen = list(res.values())[0]
        return [str(x) for x in chosen]
    elif isinstance(res, (list, set)):
        return [str(x) for x in res]
    else:
        return [str(x) for x in tokens]


def analyze_documents(
    documents: List[str],
    use_lemmatization: bool = True,
    ngram_range: Tuple[int, int] = (1, 2),
    vector_method: str = "tfidf"
) -> Dict[str, Any]:
    """
    Perform comprehensive NLP analysis on a collection of documents.

    Args:
        documents (List[str]): List of text documents to analyze.
        use_lemmatization (bool): If True, use lemmatization; else use stemming.
        ngram_range (Tuple[int, int]): Range of n-gram sizes for vectorization.
        vector_method (str): Vectorization method - "bow" or "tfidf".

    Returns:
        Dict[str, Any]: Comprehensive analysis results.
    """
    if not isinstance(documents, list):
        raise TypeError("Documents must be a list")
    if any(not isinstance(doc, str) for doc in documents):
        raise TypeError("All documents must be strings")
    if len(documents) == 0:
        raise ValueError("Documents cannot be empty")
    if len(documents) < 2:
        raise ValueError("At least 2 documents are required")
    if vector_method not in ["bow", "tfidf"]:
        raise ValueError("vector_method must be 'bow' or 'tfidf'")
    if (not isinstance(ngram_range, tuple) or len(ngram_range) != 2 or 
            ngram_range[0] > ngram_range[1] or ngram_range[0] < 1):
        raise ValueError("Invalid ngram_range")

    preprocessed_docs = []
    all_tokens = []

    for doc in documents:
        cleaned_tokens = clean_and_tokenize(doc)
        if isinstance(cleaned_tokens, str):
            cleaned_tokens = cleaned_tokens.split()
        elif not isinstance(cleaned_tokens, list):
            cleaned_tokens = list(cleaned_tokens)

        processed_tokens = _apply_stem_or_lemma(cleaned_tokens, use_lemmatization)
        processed_list = [str(t) for t in processed_tokens]
        
        preprocessed_docs.append(processed_list)
        all_tokens.extend(processed_list)

    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    avg_doc_length = total_tokens / len(documents) if documents else 0.0

    corpus = [" ".join(doc) for doc in preprocessed_docs]

    if vector_method == "bow":
        vectorizer = CountVectorizer(ngram_range=ngram_range)
    else:
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)

    X = vectorizer.fit_transform(corpus)
    feature_names = list(vectorizer.get_feature_names_out())
    doc_indices = [f"doc_{i}" for i in range(len(documents))]
    feature_df = pd.DataFrame(X.toarray(), index=doc_indices, columns=feature_names)

    w2v_sentences = [list(doc) for doc in preprocessed_docs if doc]
    if w2v_sentences:
        word2vec_model = train_word2vec(w2v_sentences)
        vocabulary_size = len(word2vec_model.wv)
        vector_size = word2vec_model.vector_size
    else:
        word2vec_model = None
        vocabulary_size = 0
        vector_size = 0

    full_text = " ".join(corpus)

    try:
        bigrams = generate_ngrams(full_text, 2) if full_text.strip() else []
        bigram_counts = count_ngrams(bigrams) if bigrams else {}
    except ValueError:
        bigrams, bigram_counts = [], {}

    try:
        trigrams = generate_ngrams(full_text, 3) if full_text.strip() else []
        trigram_counts = count_ngrams(trigrams) if trigrams else {}
    except ValueError:
        trigrams, trigram_counts = [], {}

    top_bigrams = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
    top_trigrams = sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)

    bigram_diversity = len(bigram_counts) / len(bigrams) if bigrams else 0.0

    sim_matrix = cosine_similarity(feature_df)
    similarity_df = pd.DataFrame(sim_matrix, index=doc_indices, columns=doc_indices)

    results = {
        "preprocessed": {
            "documents": preprocessed_docs,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "avg_doc_length": round(avg_doc_length, 2)
        },
        "vectorization": {
            "method": vector_method,
            "feature_matrix": feature_df,
            "feature_names": feature_names,
            "shape": feature_df.shape
        },
        "word_embeddings": {
            "model": word2vec_model,
            "vocabulary_size": vocabulary_size,
            "vector_size": vector_size
        },
        "ngram_analysis": {
            "top_bigrams": top_bigrams,
            "top_trigrams": top_trigrams,
            "bigram_diversity": round(bigram_diversity, 3)
        },
        "document_similarity": similarity_df
    }

    return results


def find_similar_documents(
    doc_index: int,
    similarity_matrix: pd.DataFrame,
    top_n: int = 3
) -> List[Tuple[int, float]]:
    """Find the most similar documents to a given document index."""
    if not isinstance(similarity_matrix, pd.DataFrame):
        raise TypeError("similarity_matrix must be a pandas DataFrame")
    if not isinstance(doc_index, int) or doc_index < 0 or doc_index >= len(similarity_matrix):
        raise ValueError(f"doc_index must be between 0 and {len(similarity_matrix) - 1}")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be at least 1")

    doc_name = similarity_matrix.index[doc_index]
    scores = similarity_matrix.loc[doc_name].drop(doc_name)
    sorted_similarities = scores.sort_values(ascending=False)

    result = []
    for target_doc, score in sorted_similarities.head(top_n).items():
        if isinstance(target_doc, (int, np.integer)):
            idx = int(target_doc)
        elif isinstance(target_doc, str) and '_' in target_doc:
            try:
                idx = int(target_doc.split('_')[1])
            except ValueError:
                idx = similarity_matrix.index.get_loc(target_doc)
        else:
            idx = similarity_matrix.index.get_loc(target_doc)
        result.append((idx, round(float(score), 4)))

    return result


def get_document_keywords(
    doc_index: int,
    feature_matrix: pd.DataFrame,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """Extract top keywords from a document based on feature scores."""
    if not isinstance(feature_matrix, pd.DataFrame):
        raise TypeError("feature_matrix must be a pandas DataFrame")
    if not isinstance(doc_index, int) or doc_index < 0 or doc_index >= len(feature_matrix):
        raise ValueError(f"doc_index must be between 0 and {len(feature_matrix) - 1}")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be at least 1")

    row = feature_matrix.iloc[doc_index]
    top_features = row.sort_values(ascending=False).head(top_n)

    result = [(str(feature), round(float(score), 4)) for feature, score in top_features.items()]

    return result


def compare_vectorization_methods(
    documents: List[str]
) -> Dict[str, pd.DataFrame]:
    """Compare BoW and TF-IDF representations side-by-side."""
    if not isinstance(documents, list):
        raise TypeError("Documents must be a list")
    if any(not isinstance(doc, str) for doc in documents):
        raise TypeError("All documents must be strings")
    if len(documents) == 0:
        raise ValueError("Documents cannot be empty")
    if len(documents) < 2:
        raise ValueError("At least 2 documents are required")

    bow_results = analyze_documents(documents, vector_method="bow")
    tfidf_results = analyze_documents(documents, vector_method="tfidf")

    return {
        "bow": bow_results['vectorization']['feature_matrix'],
        "tfidf": tfidf_results['vectorization']['feature_matrix']
    }


if __name__ == "__main__":
    sample_documents = [
        "Natural language processing is amazing and fun to learn.",
        "Machine learning and NLP are related fields in AI.",
        "Deep learning models are powerful for NLP tasks.",
        "Python is great for machine learning and data science."
    ]

    print("=" * 80)
    print("INTEGRATED NLP PIPELINE ANALYSIS")
    print("=" * 80)

    results = analyze_documents(
        sample_documents,
        use_lemmatization=True,
        ngram_range=(1, 2),
        vector_method="tfidf"
    )

    print("\n1. PREPROCESSING STATISTICS")
    print("-" * 80)
    print(f"Total tokens: {results['preprocessed']['total_tokens']}")
    print(f"Unique tokens: {results['preprocessed']['unique_tokens']}")
    print(f"Average document length: {results['preprocessed']['avg_doc_length']}")

    print("\n2. VECTORIZATION")
    print("-" * 80)
    print(f"Method: {results['vectorization']['method'].upper()}")
    print(f"Feature matrix shape: {results['vectorization']['shape']}")
    print(f"Number of features: {len(results['vectorization']['feature_names'])}")

    print("\n3. WORD EMBEDDINGS")
    print("-" * 80)
    print(f"Vocabulary size: {results['word_embeddings']['vocabulary_size']}")
    print(f"Vector dimensionality: {results['word_embeddings']['vector_size']}")

    print("\n4. N-GRAM ANALYSIS")
    print("-" * 80)
    print("Top 5 bigrams:")
    for bigram, count in results['ngram_analysis']['top_bigrams'][:5]:
        print(f"  {bigram}: {count}")
    print(f"Bigram diversity: {results['ngram_analysis']['bigram_diversity']}")

    print("\n5. DOCUMENT SIMILARITY")
    print("-" * 80)
    print("Similarity matrix:")
    print(results['document_similarity'])

    print("\n6. SIMILAR DOCUMENTS TO DOC_0")
    print("-" * 80)
    similar_docs = find_similar_documents(0, results['document_similarity'], top_n=2)
    for doc_idx, similarity in similar_docs:
        print(f"  Document {doc_idx}: {similarity:.4f}")

    print("\n7. TOP KEYWORDS IN DOC_0")
    print("-" * 80)
    keywords = get_document_keywords(0, results['vectorization']['feature_matrix'], top_n=5)
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")

    print("\n8. COMPARISON: BOW vs TF-IDF")
    print("-" * 80)
    comparison = compare_vectorization_methods(sample_documents[:3])
    print("BoW shape:", comparison['bow'].shape)
    print("TF-IDF shape:", comparison['tfidf'].shape)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)