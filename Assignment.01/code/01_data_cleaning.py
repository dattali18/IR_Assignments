# --------------------------------
# MARK: IMPORTS
# --------------------------------
import pandas as pd
import os
import re

import spacy
from spacy import Language

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from typing import Tuple

# --------------------------------
# MARK: CONSTANTS
# --------------------------------

# The data is one dir up and /data /data.xlsx
DATA_FILE_PATH = os.path.join('..', 'data', 'data.xlsx')


# --------------------------------
# MARK: DATA PREPROCESSING
# --------------------------------

def process_aj(df_aj: pd.DataFrame) -> pd.DataFrame:
    """
    @param df_aj: the df of the A-J sheet
    @rtype: pd.DataFrame
    """
    col_names_aj = ['title', 'sub_title', 'Body Text']
    # we will add all the text from the 3 column above (is nan replace by "")
    # we will add a column 'id' that will be aj_<i> where i is the index of the row

    df_aj = df_aj[col_names_aj]
    df_aj = df_aj.fillna("")

    df_aj_cpy = pd.DataFrame()
    df_aj_cpy["id"] = range(1, len(df_aj) + 1)
    df_aj_cpy["id"] = "aj_" + df_aj["id"].astype(str)
    df_aj_cpy["document"] = df_aj["title"] + " " + df_aj["sub_title"] + " " + df_aj["Body Text"]

    return df_aj_cpy


def process_bbc(df_bbc: pd.DataFrame) -> pd.DataFrame:
    """
    @param df_bbc: the df of the BBC sheet
    @return: pd.DataFrame
    """
    col_names_bbc = ['title', "Body Text"]

    df_bbc = df_bbc[col_names_bbc]
    df_bbc = df_bbc.fillna("")

    df_bbc_cpy = pd.DataFrame()
    df_bbc_cpy["id"] = range(1, len(df_bbc) + 1)
    df_bbc_cpy["id"] = "bbc_" + df_bbc['id'].astype(str)
    df_bbc_cpy["document"] = df_bbc["title"] + " " + df_bbc["Body Text"]

    return df_bbc_cpy


def process_jp(df_jp: pd.DataFrame) -> pd.DataFrame:
    """
    @param df_jp: the df of the J-P sheet
    @return: pd.DataFrame
    """
    col_names_jp = ['title', "Body"]

    df_jp = df_jp[col_names_jp]
    df_jp = df_jp.fillna("")

    df_jp_cpy = pd.DataFrame()
    df_jp_cpy["id"] = range(1, len(df_jp) + 1)
    df_jp_cpy["id"] = "jp_" + df_jp['id'].astype(str)
    df_jp_cpy["document"] = df_jp["title"] + " " + df_jp["Body"]

    return df_jp_cpy


def process_nyt(df_nyt: pd.DataFrame) -> pd.DataFrame:
    """
    @param df_nyt: the df of the NY-T sheet
    @return: pd.DataFrame
    """
    col_names_nyt = ['title', 'Body Text']

    df_nyt = df_nyt[col_names_nyt]
    df_nyt = df_nyt.fillna("")

    df_nyt_cpy = pd.DataFrame()
    df_nyt_cpy["id"] = range(1, len(df_nyt) + 1)
    df_nyt_cpy["id"] = "nyt_" + df_nyt['id'].astype(str)
    df_nyt_cpy["document"] = df_nyt["title"] + " " + df_nyt["Body Text"]

    return df_nyt_cpy


def get_excel_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_aj = pd.read_excel(path, sheet_name="A-J", engine="openpyxl")
    df_bbc = pd.read_excel(path, sheet_name="BBC", engine="openpyxl")
    df_jp = pd.read_excel(path, sheet_name="J-P", engine="openpyxl")
    df_nyt = pd.read_excel(path, sheet_name="NY-T", engine="openpyxl")

    # processing the df 1 by 1 since each df is different
    df_aj = process_aj(df_aj)
    df_bbc = process_bbc(df_bbc)
    df_jp = process_jp(df_jp)
    df_nyt = process_nyt(df_nyt)

    return df_aj, df_bbc, df_jp, df_nyt


# --------------------------------
# MARK: DATA CLEANING - PART 1 WORDS
# --------------------------------

def clean_text(text: str) -> str:
    """
    @summary Clean the text by normalizing all types of single and double quotation marks to standard forms
    @param text: the text to clean
    @rtype: str
    """
    # Normalize all types of single and double quotation marks to standard forms
    text = re.sub(r"[‘’`]", "'", text)  # Convert all single quote variations to '
    text = re.sub(r"[“”]", '"', text)  # Convert all double quote variations to "

    return text


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    @summary Clean the text in the dataframe
    @param df: the dataframe to clean
    @rtype: pd.DataFrame
    """
    df["document"] = df["document"].apply(clean_text)

    return df


def clean_word(text: str) -> str:
    # Tokenize with regex to handle punctuation outside of words and contractions
    tokens = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text, re.UNICODE)

    return ' '.join(tokens)


def clean_word_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    @summary Clean the words in the dataframe
    @param df: the dataframe to clean
    @rtype: pd.DataFrame
    """
    df["document"] = df["document"].apply(clean_word)

    return df


def save_documents_to_csv(df: pd.DataFrame, file_path: str) -> None:
    df.to_csv(file_path, index=False)


# --------------------------------
# MARK: DATA CLEANING - PART 2 LEMMATIZATION
# --------------------------------

def download_nltk_resources():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download('punkt_tab')


def get_stop_words() -> Tuple[set, Language]:
    # before loading the stop words, we need to download the resources
    # you need to run: `python -m spacy download en_core_web_sm` before running this function
    nlp = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words("english"))

    return stop_words, nlp


def clean_lemma(text: str) -> str:
    stop_words, nlp = get_stop_words()

    # replace all the `I'm` to I am etc
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"i'll", "I will", text)

    # remove from the text all the punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # tokenize the text
    tokens = word_tokenize(text)

    # remove all the numbers and dates etc
    tokens = [word for word in tokens if not any(char.isdigit() for char in word)]

    # remove the stopwords
    tokens = [word for word in tokens if not word.lower in stop_words]

    doc = nlp(' '.join(tokens))
    legitimatized_text = ' '.join(token.lemma_ for token in doc)

    return legitimatized_text


def clean_lemma_df(df):
    df_copy = df.copy()
    df_copy["document"] = df_copy["document"].astype(str).apply(clean_lemma)
    return df_copy


# --------------------------------
# MARK: MAIN
# --------------------------------

def main():
    # PART 0: Preprocessing
    # Get the data from the Excel file
    df_aj, df_bbc, df_jp, df_nyt = get_excel_data(DATA_FILE_PATH)

    # Clean the text in the dataframes
    df_aj = clean_df(df_aj)
    df_bbc = clean_df(df_bbc)
    df_jp = clean_df(df_jp)
    df_nyt = clean_df(df_nyt)

    # PART 1: WORDS
    # Clean the words in the dataframes
    df_aj = clean_word_df(df_aj)
    df_bbc = clean_word_df(df_bbc)
    df_jp = clean_word_df(df_jp)
    df_nyt = clean_word_df(df_nyt)

    # Save the dataframes to csv files
    data_word_folder = os.path.join('..', 'data', 'word')

    save_documents_to_csv(df_aj, os.path.join(data_word_folder, 'aj.csv'))
    save_documents_to_csv(df_bbc, os.path.join(data_word_folder, 'bbc.csv'))
    save_documents_to_csv(df_jp, os.path.join(data_word_folder, 'jp.csv'))
    save_documents_to_csv(df_nyt, os.path.join(data_word_folder, 'nyt.csv'))

    # PART 2: LEMMATIZATION
    # Download the necessary resources from NLTK
    download_nltk_resources()

    # Clean the lemmas in the dataframes
    df_aj_lemma = clean_lemma_df(df_aj)
    df_bbc_lemma = clean_lemma_df(df_bbc)
    df_jp_lemma = clean_lemma_df(df_jp)
    df_nyt_lemma = clean_lemma_df(df_nyt)

    # Save the dataframes to csv files
    data_lemma_folder = os.path.join('..', 'data', 'lemma')

    save_documents_to_csv(df_aj_lemma, os.path.join(data_lemma_folder, 'aj.csv'))
    save_documents_to_csv(df_bbc_lemma, os.path.join(data_lemma_folder, 'bbc.csv'))
    save_documents_to_csv(df_jp_lemma, os.path.join(data_lemma_folder, 'jp.csv'))
    save_documents_to_csv(df_nyt_lemma, os.path.join(data_lemma_folder, 'nyt.csv'))


if __name__ == '__main__':
    main()
