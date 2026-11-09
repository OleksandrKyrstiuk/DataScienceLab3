"""Task 4: TF-IDF Calculation."""
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_tfidf(texts: List[str]) -> pd.DataFrame:
    """
    Convert a list of text strings into a TF-IDF representation.

    Args:
        texts (List[str]): A list of text strings.

    Returns:
        pd.DataFrame: A DataFrame representing the TF-IDF model,
                      with words as columns and their TF-IDF scores as values.

    Raises:
        TypeError: If texts is not a list.
        ValueError: If texts is empty or contains non-string elements.

    Example:
        >>> texts = ["I love NLP", "NLP is fun"]
        >>> calculate_tfidf(texts)
                fun         is       love        nlp
        0  0.000000   0.000000   0.795961   0.605349
        1  0.795961   0.795961   0.000000   0.605349
    """
    # 1. Validation for non-list input
    if not isinstance(texts, list):
        raise TypeError("Input texts must be a list")

    # 2. Validation for empty list
    if not texts:
        raise ValueError("Input texts cannot be empty")

    # 3. Validation for non-string elements inside list
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("All texts must be strings")

    # 4. Initialize TfidfVectorizer and compute TF-IDF matrix
    vectorizer = TfidfVectorizer()
    sparse_matrix = vectorizer.fit_transform(texts)

    # 5. Build DataFrame with feature names (words) as columns
    df = pd.DataFrame(
        sparse_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    return df


if __name__ == "__main__":
    texts = [
        "Converts a list of text strings into a Bag of Words representation",
        "The words that changed the strings",
        "Your duty words of the heart",
        "The Bag full of strings"
    ]
    print(calculate_tfidf(texts))