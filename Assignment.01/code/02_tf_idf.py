# -----------------------------------------------
# DESCRIPTION
# -----------------------------------------------
# This script will take the cleaned data from the
# last section and create the TF-IDF matrix using
# the Okapi BM25 algorithms.
# We will create a matrix for each group of
# document, we have a total of 8 groups each
# group as about 600 document.
# -----------------------------------------------

# -----------------------------------------------
# IMPORTS
# -----------------------------------------------
import pandas as pd
import re
import os

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from scipy import sparse

# -----------------------------------------------
# GLOBALS
# -----------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
WORD_DIR = os.path.join(DATA_DIR, "words")
LEMMA_DIR = os.path.join(DATA_DIR, "lemma")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data", "matrix")

STOP_WORDS = set(stopwords.words("english"))

# -----------------------------------------------
# FUNCTIONS
# -----------------------------------------------


def load_data(path):
    return pd.read_csv(path)


def create_corpus(corpus):
    processed_corpus = []

    for doc in corpus:
        # Convert to lowercase
        doc = doc.lower()
        # Remove punctuation
        doc = re.sub(r"[^\w\s]", "", doc)
        # Remove numbers
        doc = re.sub(r"\d+", "", doc)
        # Tokenize the document
        words = word_tokenize(doc)
        # Remove stop words
        words = [word for word in words if word not in STOP_WORDS]
        # Join the words back into a string
        processed_doc = " ".join(words)
        processed_corpus.append(processed_doc)
    return processed_corpus


def create_vocabulary(corpus):
    vocabulary = set()
    for doc in corpus:
        words = doc.split()
        for word in words:
            vocabulary.add(word)
    return vocabulary


def create_tf_matrix(corpus, vocab, threshold=5):
    n_docs = len(corpus)
    n_terms = len(vocab)

    tf_matrix = np.zeros((n_docs, n_terms)).astype(np.int32)

    for doc_idx, doc in enumerate(corpus):
        words = doc.split()

        for v_idx, v in enumerate(vocab):
            # if the count is lower than threshold don't put it in
            c = words.count(v)
            if c >= threshold:
                tf_matrix[doc_idx, v_idx] = words.count(v)

    return tf_matrix


# perform the tf-idf bm25 okapi algorithm on the data
def calculate_df(tf_matrix, vocab):
    df = {}
    for term_idx in range(tf_matrix.shape[1]):
        df[vocab[term_idx]] = np.count_nonzero(tf_matrix[:, term_idx])
    return df


def calculate_avg_doc_len(corpus):
    total_len = sum(len(doc.split()) for doc in corpus)
    return total_len / len(corpus)


def tfidf_bm25_okapi(tf_matrix, df, processed_corpus, vocab, L_avg, k=1.2, b=0.75):
    M = tf_matrix.shape[0]  # Total number of documents

    tfidf_matrix = np.zeros_like(tf_matrix, dtype=np.float64)

    for doc_idx in range(tf_matrix.shape[0]):
        doc_len = len(processed_corpus[doc_idx].split())
        for term_idx in range(tf_matrix.shape[1]):
            term = vocab[term_idx]
            c_wd = tf_matrix[doc_idx, term_idx]  # Term frequency in the document

            if c_wd > 0:  # Only calculate if the term is present
                idf = np.log((M + 1) / df[term])

                numerator = (k + 1) * c_wd
                denominator = c_wd + k * (1 - b + b * (doc_len / L_avg))

                tfidf_matrix[doc_idx, term_idx] = (numerator / denominator) * idf

    return tfidf_matrix


# -----------------------------------------------
# SAVING THE DATA
# -----------------------------------------------
# We will save the data in the following way:
# 1. the tf-idf matrix will be save in a sparse
#    matrix using the scipy library
# 2. the vocabulary will be save in a csv file
# 3. the document id will be save in a csv file
#
# We will also write a function the will load
# the data and transform it into a pandas dataframe.
# -----------------------------------------------


def save_sparse_matrix(matrix, vocab, doc_ids, path):
    # convert the matrix to a sparse matrix
    sparse_matrix = sparse.csr_matrix(matrix)
    # save the sparse matrix
    sparse.save_npz(path + "_matrix.npz", sparse_matrix)
    # save the vocabulary
    pd.DataFrame(list(vocab), columns=["term"]).to_csv(path + "_vocab.csv", index=False)
    # save the document id
    pd.DataFrame(doc_ids, columns=["doc_id"]).to_csv(path + "_doc_id.csv", index=False)


def load_sparse_matrix(path):
    # load the sparse matrix
    matrix = sparse.load_npz(path + "_matrix.npz")
    # load the vocabulary
    vocab = pd.read_csv(path + "_vocab.csv")["term"].tolist()
    # load the document id
    doc_ids = pd.read_csv(path + "_doc_id.csv")["doc_id"].tolist()

    # transform the sparse matrix to a dense matrix and create a df with vocab and doc_id
    matrix = matrix.todense()
    df = pd.DataFrame(matrix, columns=vocab, index=doc_ids)
    return df, matrix, vocab, doc_ids


# -----------------------------------------------
# MAIN
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

    # create a corpus from each data group
    corpus_aj_word = create_corpus(df_aj_word["document"].tolist())
    corpus_bbc_word = create_corpus(df_bbc_word["document"].tolist())
    corpus_jp_word = create_corpus(df_jp_word["document"].tolist())
    corpus_nyt_word = create_corpus(df_nyt_word["document"].tolist())

    corpus_aj_lemma = create_corpus(df_aj_lemma["document"].tolist())
    corpus_bbc_lemma = create_corpus(df_bbc_lemma["document"].tolist())
    corpus_jp_lemma = create_corpus(df_jp_lemma["document"].tolist())
    corpus_nyt_lemma = create_corpus(df_nyt_lemma["document"].tolist())

    # create a vocabulary from each data group
    vocabulary_aj_word = create_vocabulary(corpus_aj_word)
    vocabulary_bbc_word = create_vocabulary(corpus_bbc_word)
    vocabulary_jp_word = create_vocabulary(corpus_jp_word)
    vocabulary_nyt_word = create_vocabulary(corpus_nyt_word)

    vocabulary_aj_lemma = create_vocabulary(corpus_aj_lemma)
    vocabulary_bbc_lemma = create_vocabulary(corpus_bbc_lemma)
    vocabulary_jp_lemma = create_vocabulary(corpus_jp_lemma)
    vocabulary_nyt_lemma = create_vocabulary(corpus_nyt_lemma)

    # create the TF matrix for each data group
    tf_matrix_aj_word = create_tf_matrix(corpus_aj_word, vocabulary_aj_word)
    tf_matrix_bbc_word = create_tf_matrix(corpus_bbc_word, vocabulary_bbc_word)
    tf_matrix_jp_word = create_tf_matrix(corpus_jp_word, vocabulary_jp_word)
    tf_matrix_nyt_word = create_tf_matrix(corpus_nyt_word, vocabulary_nyt_word)

    tf_matrix_aj_lemma = create_tf_matrix(corpus_aj_lemma, vocabulary_aj_lemma)
    tf_matrix_bbc_lemma = create_tf_matrix(corpus_bbc_lemma, vocabulary_bbc_lemma)
    tf_matrix_jp_lemma = create_tf_matrix(corpus_jp_lemma, vocabulary_jp_lemma)
    tf_matrix_nyt_lemma = create_tf_matrix(corpus_nyt_lemma, vocabulary_nyt_lemma)

    # calculate the document frequency for each data group
    # dfc = document frequency count
    dfc_aj_word = calculate_df(tf_matrix_aj_word, vocabulary_aj_word)
    dfc_bbc_word = calculate_df(tf_matrix_bbc_word, vocabulary_bbc_word)
    dfc_jp_word = calculate_df(tf_matrix_jp_word, vocabulary_jp_word)
    dfc_nyt_word = calculate_df(tf_matrix_nyt_word, vocabulary_nyt_word)

    dfc_aj_lemma = calculate_df(tf_matrix_aj_lemma, vocabulary_aj_lemma)
    dfc_bbc_lemma = calculate_df(tf_matrix_bbc_lemma, vocabulary_bbc_lemma)
    dfc_jp_lemma = calculate_df(tf_matrix_jp_lemma, vocabulary_jp_lemma)
    dfc_nyt_lemma = calculate_df(tf_matrix_nyt_lemma, vocabulary_nyt_lemma)

    # calculate the average document length for each data group
    L_avg_aj_word = calculate_avg_doc_len(corpus_aj_word)
    L_avg_bbc_word = calculate_avg_doc_len(corpus_bbc_word)
    L_avg_jp_word = calculate_avg_doc_len(corpus_jp_word)
    L_avg_nyt_word = calculate_avg_doc_len(corpus_nyt_word)

    L_avg_aj_lemma = calculate_avg_doc_len(corpus_aj_lemma)
    L_avg_bbc_lemma = calculate_avg_doc_len(corpus_bbc_lemma)
    L_avg_jp_lemma = calculate_avg_doc_len(corpus_jp_lemma)
    L_avg_nyt_lemma = calculate_avg_doc_len(corpus_nyt_lemma)

    # calculate the tf-idf matrix for each data group
    tfidf_aj_word = tfidf_bm25_okapi(
        tf_matrix_aj_word,
        dfc_aj_word,
        corpus_aj_word,
        vocabulary_aj_word,
        L_avg_aj_word,
    )
    tfidf_bbc_word = tfidf_bm25_okapi(
        tf_matrix_bbc_word,
        dfc_bbc_word,
        corpus_bbc_word,
        vocabulary_bbc_word,
        L_avg_bbc_word,
    )
    tfidf_jp_word = tfidf_bm25_okapi(
        tf_matrix_jp_word,
        dfc_jp_word,
        corpus_jp_word,
        vocabulary_jp_word,
        L_avg_jp_word,
    )
    tfidf_nyt_word = tfidf_bm25_okapi(
        tf_matrix_nyt_word,
        dfc_nyt_word,
        corpus_nyt_word,
        vocabulary_nyt_word,
        L_avg_nyt_word,
    )

    tfidf_aj_lemma = tfidf_bm25_okapi(
        tf_matrix_aj_lemma,
        dfc_aj_lemma,
        corpus_aj_lemma,
        vocabulary_aj_lemma,
        L_avg_aj_lemma,
    )
    tfidf_bbc_lemma = tfidf_bm25_okapi(
        tf_matrix_bbc_lemma,
        dfc_bbc_lemma,
        corpus_bbc_lemma,
        vocabulary_bbc_lemma,
        L_avg_bbc_lemma,
    )
    tfidf_jp_lemma = tfidf_bm25_okapi(
        tf_matrix_jp_lemma,
        dfc_jp_lemma,
        corpus_jp_lemma,
        vocabulary_jp_lemma,
        L_avg_jp_lemma,
    )
    tfidf_nyt_lemma = tfidf_bm25_okapi(
        tf_matrix_nyt_lemma,
        dfc_nyt_lemma,
        corpus_nyt_lemma,
        vocabulary_nyt_lemma,
        L_avg_nyt_lemma,
    )

    # save the data
    save_sparse_matrix(
        tfidf_aj_word,
        vocabulary_aj_word,
        df_aj_word["doc_id"].tolist(),
        os.join(OUTPUT_DIR, "aj_word"),
    )
    save_sparse_matrix(
        tfidf_bbc_word,
        vocabulary_bbc_word,
        df_bbc_word["doc_id"].tolist(),
        os.join(OUTPUT_DIR, "bbc_word"),
    )
    save_sparse_matrix(
        tfidf_jp_word,
        vocabulary_jp_word,
        df_jp_word["doc_id"].tolist(),
        os.join(OUTPUT_DIR, "jp_word"),
    )
    save_sparse_matrix(
        tfidf_nyt_word,
        vocabulary_nyt_word,
        df_nyt_word["doc_id"].tolist(),
        os.join(OUTPUT_DIR, "nyt_word"),
    )

    save_sparse_matrix(
        tfidf_aj_lemma,
        vocabulary_aj_lemma,
        df_aj_lemma["doc_id"].tolist(),
        os.join(OUTPUT_DIR, "aj_lemma"),
    )
    save_sparse_matrix(
        tfidf_bbc_lemma,
        vocabulary_bbc_lemma,
        df_bbc_lemma["doc_id"].tolist(),
        os.join(OUTPUT_DIR, "bbc_lemma"),
    )

    save_sparse_matrix(
        tfidf_jp_lemma,
        vocabulary_jp_lemma,
        df_jp_lemma["doc_id"].tolist(),
        os.join(OUTPUT_DIR, "jp_lemma"),
    )
    save_sparse_matrix(
        tfidf_nyt_lemma,
        vocabulary_nyt_lemma,
        df_nyt_lemma["doc_id"].tolist(),
        os.join(OUTPUT_DIR, "nyt_lemma"),
    )


if __name__ == "__main__":
    main()
