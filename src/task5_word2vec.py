"""Task 5: Word2Vec Model Training."""
from typing import List, Tuple
from gensim.models import Word2Vec


def train_word2vec(
    sentences: List[List[str]], 
    vector_size: int = 100, 
    window: int = 5, 
    min_count: int = 1, 
    workers: int = 4
) -> Word2Vec:
    """
    Train a Word2Vec model on a list of tokenized sentences.

    Args:
        sentences (List[List[str]]): A list of tokenized sentences.
        vector_size (int): Dimensionality of the word vectors. Default is 100.
        window (int): Maximum distance between current and predicted word. Default is 5.
        min_count (int): Ignores words with frequency lower than this. Default is 1.
        workers (int): Number of worker threads for training. Default is 4.

    Returns:
        Word2Vec: A trained Word2Vec model.

    Raises:
        TypeError: If sentences is not a list of lists.
        ValueError: If sentences is empty or contains invalid data.

    Example:
        >>> sentences = [['i', 'love', 'nlp'], ['nlp', 'is', 'fun']]
        >>> model = train_word2vec(sentences)
        >>> 'nlp' in model.wv
        True
    """

    if not isinstance(sentences, list):
        raise TypeError("Sentences must be a list")

    if not sentences:
        raise ValueError("Sentences cannot be empty")

    for sentence in sentences:
        if not isinstance(sentence, list):
            raise TypeError("Sentences must be a list of lists")
        if not sentence:
            raise ValueError("Sentences cannot contain empty lists")
        if any(not isinstance(word, str) for word in sentence):
            raise ValueError("All tokens in sentences must be strings")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    return model


def find_similar_words(model: Word2Vec, word: str, topn: int = 10) -> List[Tuple[str, float]]:
    """
    Find the most similar words to a given word using a trained Word2Vec model.

    Args:
        model (Word2Vec): A trained Word2Vec model.
        word (str): The word to find similar words for.
        topn (int): Number of top similar words to return. Default is 10.

    Returns:
        List[Tuple[str, float]]: A list of tuples with similar words and their 
                                 similarity scores.

    Raises:
        KeyError: If the word is not in the model's vocabulary.
        TypeError: If model is not a Word2Vec model or word is not a string.

    Example:
        >>> sentences = [['i', 'love', 'nlp'], ['nlp', 'is', 'fun']]
        >>> model = train_word2vec(sentences)
        >>> similar = find_similar_words(model, 'nlp', topn=2)
        >>> len(similar) <= 2
        True
    """

    if not isinstance(model, Word2Vec):
        raise TypeError("Model must be an instance of Word2Vec")
    if not isinstance(word, str):
        raise TypeError("Word must be a string")

    if word not in model.wv:
        raise KeyError(f"Word '{word}' is not in vocabulary")

    return model.wv.most_similar(word, topn=topn)


if __name__ == "__main__":
    import nltk
    nltk.download('punkt', quiet=True)

    example_sentences = [
        ['i', 'love', 'nlp'],
        ['nlp', 'is', 'fun'],
        ['deep', 'learning', 'is', 'a', 'subset', 'of', 'machine', 'learning'],
        ['machine', 'learning', 'is', 'fun'],
        ['nlp', 'can', 'be', 'challenging'],
        ['artificial', 'intelligence', 'is', 'related', 'to', 'nlp']
    ]

    model = train_word2vec(example_sentences)

    similar_words = find_similar_words(model, 'nlp')
    print("Words similar to 'nlp':")
    for word, score in similar_words:
        print(f"  {word}: {score:.4f}")

    similar_words = find_similar_words(model, 'learning')
    print("\nWords similar to 'learning':")
    for word, score in similar_words:
        print(f"  {word}: {score:.4f}")