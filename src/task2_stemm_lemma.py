"""Task 2: Stemming and Lemmatization."""
from typing import Dict, List
from nltk.stem import PorterStemmer, WordNetLemmatizer


def stem_and_lemmatize(tokens: List[str]) -> Dict[str, List[str]]:
    """
    Apply stemming and lemmatization to input tokens.

    Args:
        tokens (List[str]): A list of tokens (words).

    Returns:
        Dict[str, List[str]]: A dictionary with 'original', 'stemmed', and 
                              'lemmatized' token lists.

    Raises:
        TypeError: If tokens is not a list.
        ValueError: If tokens is empty or contains non-string elements.

    Example:
        >>> stem_and_lemmatize(['running', 'cats'])
        {'original': ['running', 'cats'], 'stemmed': ['run', 'cat'], 
         'lemmatized': ['running', 'cat']}
    """
    # 1. Validation for non-list input
    if not isinstance(tokens, list):
        raise TypeError("Input tokens must be a list")

    # 2. Validation for empty list
    if not tokens:
        raise ValueError("Input tokens cannot be empty")

    # 3. Validation for non-string elements
    if any(not isinstance(token, str) for token in tokens):
        raise ValueError("All tokens must be strings")

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stemmed = [stemmer.stem(token) for token in tokens]
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]

    return {
        "original": tokens,
        "stemmed": stemmed,
        "lemmatized": lemmatized
    }


if __name__ == "__main__":
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

    example_tokens = ['goes', 'worlds', 'done', 'being', 'undefined']
    result = stem_and_lemmatize(example_tokens)
    print(f"Original: {result['original']}")
    print(f"Stemmed: {result['stemmed']}")
    print(f"Lemmatized: {result['lemmatized']}")