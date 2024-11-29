# -----------------------------------------------
# Imports
# -----------------------------------------------
import pandas as pd
import numpy as np
import os
import re

# NLP
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Vectorization
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel

# -----------------------------------------------
# GLOBALS
# -----------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
WORD_DIR = os.path.join(DATA_DIR, "words")
LEMMA_DIR = os.path.join(DATA_DIR, "lemma")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data", "vectors")

STOP_WORDS = set(stopwords.words("english"))

# -----------------------------------------------
# Functions
# -----------------------------------------------


def load_data(path):
    return pd.read_csv(path)


def clean_text(text):
    # tokenize the text
    tokens = word_tokenize(text, language="english")

    # tokens is a list of str remove duplicate so it's easier to work with

    # remove the stop words
    stop_words = STOP_WORDS
    tokens = [word for word in tokens if word not in stop_words]

    # transform all the word into lower case
    tokens = [word.lower() for word in tokens]

    # remove the punctuation using re
    tokens = [word for word in tokens if re.match(r"[a-zA-Z]+", word)]

    return tokens


def vectorize_words(model, tokens):
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
    return vectors


def create_matrix(vectors):
    matrix = np.array(vectors)
    return matrix


def create_document_vector(matrix):
    # the matrix is a list of list of number
    np_matrix = np.array(matrix)
    return np.mean(np_matrix, axis=0)


def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = (
        outputs.last_hidden_state[:, 0, :].detach().numpy()
    )  # Use the [CLS] token embedding
    return embeddings[0]


# -----------------------------------------------
# Main
# -----------------------------------------------

def main():
    # Load the data
    # setting the path to the data
    # 1. word files
    aj_word_file = os.path.join(WORD_DIR, "A_J_word.csv")
    bbc_word_file = os.path.join(WORD_DIR, "BBC_word.csv")
    jp_word_file = os.path.join(WORD_DIR, "J_P_word.csv")
    nyt_word_file = os.path.join(WORD_DIR, "NYT_word.csv")

    # 2. lemma files
    aj_lemma_file = os.path.join(LEMMA_DIR, "A_J_lemma.csv")
    bbc_lemma_file = os.path.join(LEMMA_DIR, "BBC_lemma.csv")
    jp_lemma_file = os.path.join(LEMMA_DIR, "J_P_lemma.csv")
    nyt_lemma_file = os.path.join(LEMMA_DIR, "NYT_lemma.csv")

    # loading the data
    df_aj_word = load_data(aj_word_file)
    df_bbc_word = load_data(bbc_word_file)
    df_jp_word = load_data(jp_word_file)
    df_nyt_word = load_data(nyt_word_file)

    df_aj_lemma = load_data(aj_lemma_file)
    df_bbc_lemma = load_data(bbc_lemma_file)
    df_jp_lemma = load_data(jp_lemma_file)
    df_nyt_lemma = load_data(nyt_lemma_file)

    #  we will do it for one example and then for all 2 * 4 documents corpus
    df = df_aj_word

    df['tokens'] = df['text'].apply(clean_text)
    # get the length of the tokens
    df['len_tokens'] = df['tokens'].apply(len)

    # getting the vectors
    model = Word2Vec(df["tokens"], min_count=5)

    df["vectors"] = df["tokens"].apply(vectorize_words, model=model)

    # create the matrix
    df["matrix"] = df["vectors"].apply(create_matrix).tolist()

    # save the matrix
    # todo: later

    # document matrix
    df["document_vector"] = df["matrix"].apply(create_document_vector)

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # BERT Embedding for each word in each document
    # The words are already tokenized in 'tokens' column
    df["bert_word_embedding"] = df["tokens"].apply(
        lambda x: [get_bert_embedding(word, tokenizer=tokenizer, model=model) for word in x]
    )

    # BERT Embedding for the all document
    df["bert_embedding"] = df["document"].apply(get_bert_embedding, tokenizer=tokenizer, model=model)

    # Save the data

    # to complete the assignment do all that for each of the remaining 7 document groups

if __name__ == "__main__":
    main()
