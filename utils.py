import re
from gensim.parsing.preprocessing import remove_stopwords


def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # Remove small words
    sentence = re.sub(r'\W*\b\w{1,2}\b', ' ', sentence)

    # Removing stopwords with gensim
    sentence = remove_stopwords(sentence.lower())

    return sentence


