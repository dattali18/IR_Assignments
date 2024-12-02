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


def information_gain(tfidf_matrix):
    # calculate the entropy of the tfidf matrix
    entropy = -tfidf_matrix * np.log2(tfidf_matrix)
    entropy = np.sum(entropy, axis=1)
    # calculate the entropy of the tfidf matrix
    entropy = -tfidf_matrix * np.log2(tfidf_matrix)
    entropy = np.sum(entropy, axis=1)
    # calculate the information gain
    ig = np.sum(entropy) - entropy
    return ig


def gain_ration(tfidf_matrix):
    """
    DESCRIPTION
    -----------
    Calculate the gain ratio of the tfidf matrix
    gain ratio = information gain / entropy of the tfidf matrix
    the gain ration is used to determine the importance of a term
    """
    # calculate the entropy of the tfidf matrix
    entropy = -tfidf_matrix * np.log2(tfidf_matrix)
    entropy = np.sum(entropy, axis=1)
    # calculate the information gain
    ig = np.sum(entropy) - entropy
    # calculate the gain ratio
    gr = ig / entropy
    return gr


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
    word_file = [aj_word_file, bbc_word_file, jp_word_file, nyt_word_file]
    lemma_file = [aj_lemma_file, bbc_lemma_file, jp_lemma_file, nyt_lemma_file]

    # transform the line by line into a dict comprehension
    dfs_word = {file: load_data(file) for file in word_file}
    dfs_lemma = {file: load_data(file) for file in lemma_file}

    # create a corpus from each data group

    corpuses_word = {
        file: create_corpus(df["document"].tolist()) for file, df in dfs_word.items()
    }
    corpuses_lemma = {
        file: create_corpus(df["document"].tolist()) for file, df in dfs_lemma.items()
    }

    # create a vocabulary from each data group
    vocabularies_word = {
        file: create_vocabulary(corpus) for file, corpus in corpuses_word.items()
    }
    vocabularies_lemma = {
        file: create_vocabulary(corpus) for file, corpus in corpuses_lemma.items()
    }

    # create the TF matrix for each data group
    tf_matrices_word = {
        file: create_tf_matrix(corpus, vocab)
        for file, (corpus, vocab) in zip(
            corpuses_word.items(), vocabularies_word.items()
        )
    }
    tf_matrices_lemma = {
        file: create_tf_matrix(corpus, vocab)
        for file, (corpus, vocab) in zip(
            corpuses_lemma.items(), vocabularies_lemma.items()
        )
    }

    # calculate the document frequency for each data group
    # dfc = document frequency count
    dfcs_word = {
        file: calculate_df(tf_matrix, vocab)
        for file, (tf_matrix, vocab) in zip(
            tf_matrices_word.items(), vocabularies_word.items()
        )
    }
    dfcs_lemma = {
        file: calculate_df(tf_matrix, vocab)
        for file, (tf_matrix, vocab) in zip(
            tf_matrices_lemma.items(), vocabularies_lemma.items()
        )
    }

    # calculate the average document length for each data group
    L_avgs_word = {
        file: calculate_avg_doc_len(corpus) for file, corpus in corpuses_word.items()
    }
    L_avgs_lemma = {
        file: calculate_avg_doc_len(corpus) for file, corpus in corpuses_lemma.items()
    }

    # calculate the tf-idf matrix for each data group
    tfidf_matrices_word = {
        file: tfidf_bm25_okapi(tf_matrix, dfc, corpus, vocab, L_avg)
        for file, (tf_matrix, dfc, corpus, vocab, L_avg) in zip(
            tf_matrices_word.items(),
            dfcs_word.items(),
            corpuses_word.items(),
            vocabularies_word.items(),
            L_avgs_word.items(),
        )
    }

    tfidf_matrices_lemma = {
        file: tfidf_bm25_okapi(tf_matrix, dfc, corpus, vocab, L_avg)
        for file, (tf_matrix, dfc, corpus, vocab, L_avg) in zip(
            tf_matrices_lemma.items(),
            dfcs_lemma.items(),
            corpuses_lemma.items(),
            vocabularies_lemma.items(),
            L_avgs_lemma.items(),
        )
    }

    # save all the data
    for file, tfidf_matrix in tfidf_matrices_word.items():
        vocabulary = vocabularies_word[file]
        doc_ids = dfs_word[file]["doc_id"].tolist()
        save_sparse_matrix(tfidf_matrix, vocabulary, doc_ids, file + "_word")

    # Perform the data analysis on the tf-idf matrix
    # 1. Information Gain

    ig_word = {  # information gain for the word data
        file: information_gain(tfidf_matrix)
        for file, tfidf_matrix in tfidf_matrices_word.items()
    }

    ig_lemma = {  # information gain for the lemma data
        file: information_gain(tfidf_matrix)
        for file, tfidf_matrix in tfidf_matrices_lemma.items()
    }
    # 2. Gain Ratio

    gr_word = {  # gain ratio for the word data
        file: gain_ration(tfidf_matrix)
        for file, tfidf_matrix in tfidf_matrices_word.items()
    }

    gr_lemma = {  # gain ratio for the lemma data
        file: gain_ration(tfidf_matrix)
        for file, tfidf_matrix in tfidf_matrices_lemma.items()
    }

    # print the most important terms for each technique
    # for each group find the term with the highest information gain
    for file, ig in ig_word.items():
        print(
            f"Most important term for {file} using information gain: {vocabularies_word[file][np.argmax(ig)]}"
        )

    for file, ig in ig_lemma.items():
        print(
            f"Most important term for {file} using information gain: {vocabularies_lemma[file][np.argmax(ig)]}"
        )

    # for each group find the term with the highest gain ratio
    for file, gr in gr_word.items():
        print(
            f"Most important term for {file} using gain ratio: {vocabularies_word[file][np.argmax(gr)]}"
        )

    for file, gr in gr_lemma.items():
        print(
            f"Most important term for {file} using gain ratio: {vocabularies_lemma[file][np.argmax(gr)]}"
        )


if __name__ == "__main__":
    main()
