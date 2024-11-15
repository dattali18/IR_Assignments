"""
# Information Retrival Course

## Assigment 1

### Step 1: Data Cleaning
"""

# cleaning the data
#!rm data.xlsx

# downloading the data
import requests
import os
import pandas as pd

url = "https://github.com/dattali18/IR_Assignments/blob/main/Assignment.01/data/data.xlsx?raw=true"
output_filename = "data.xlsx"

response = requests.get(url)
if response.status_code == 200:
    with open(output_filename, "wb") as file:
        file.write(response.content)

print(f"The file was downloaded, and it is in {output_filename}.")

data_aj = pd.read_excel(output_filename, sheet_name="A-J", engine="openpyxl")
data_bbc = pd.read_excel(output_filename, sheet_name="BBC", engine="openpyxl")
data_jp = pd.read_excel(output_filename, sheet_name="J-P", engine="openpyxl")
data_nyt = pd.read_excel(output_filename, sheet_name="NY-T", engine="openpyxl")

text_column_name = "Body Text"
title_column_name = "title"

# crate a df from each sourcre with only the "Body Text" and "title" and rename the colmun into "text" and "title"
df_aj = data_aj[[text_column_name, title_column_name]].rename(columns={text_column_name: "text", title_column_name: "title"})

# print the header of the df to see if sucessful

df_aj.head()

# do the same for all other

df_bbc = data_bbc[[text_column_name, title_column_name]].rename(columns={text_column_name: "text", title_column_name: "title"})
df_jp = data_jp[["Body", title_column_name]].rename(columns={"Body": "text", title_column_name: "title"})
df_nyt = data_nyt[[text_column_name, title_column_name]].rename(columns={text_column_name: "text", title_column_name: "title"})

"""Now we have 4 df with the column of text and title for each one of the 4 source and each data frame has about 600 articles."""

print(f"Article from A_J: {len(df_aj)}")
print(f"Article from BBC: {len(df_bbc)}")
print(f"Article from J_P: {len(df_jp)}")
print(f"Article from NYT: {len(df_nyt)}")

"""Now we will perform the data cleaning in two steps:

1. cleaning the word from all the punctuation marks.
  - for example "How are you?" -> "How are you ?" because "you" != "you?"
2. cleaning the documents with lemmatisation.
 - "cleaning" -> "clean"

 Outpute at the end:

 1. 4 cleaned document sets from the ponctuation (not deleting them).
 2. 4 cleaned document set by that include only the lemma (not word but their root).
"""

# performing the cleaning number 1
import re
import pandas as pd

def clean_text(text):
    # Normalize all types of single and double quotation marks to standard forms
    text = re.sub(r"[‘’`]", "'", text)  # Convert all single quote variations to '
    text = re.sub(r"[“”]", '"', text)   # Convert all double quote variations to "

    # Tokenize with regex to handle punctuation outside of words and contractions
    tokens = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text, re.UNICODE)

    return ' '.join(tokens)

# test the cleaning with a df
# we can perform more tests as we find extreme cases.

text = "How are you? I'm well thank you. doesn't."
clean_text(text)

# let's write the data cleaning


def clean_df_ponctuation(df):
  """
  This function should return a new df and not change the old one.
  """
  df_copy = df.copy()
  df_copy["text"] = df_copy["text"].astype(str).apply(clean_text)
  return df_copy

# test the function with a data frame and print the
df_aj_ponctuation = clean_df_ponctuation(df_aj)

# test the text that was produce

for i in range(5):
  print(f"{i + 1}: {data_aj_ponctuation['text'][i]}")

# clean all the other df

df_bbc_ponctuation = clean_df_ponctuation(df_bbc)
df_jp_ponctuation = clean_df_ponctuation(df_jp)
df_nyt_ponctuation = clean_df_ponctuation(df_nyt)

"""Now that we have clean the data in the first type we will do the lemmatisation."""

# installing the needed packeges
#!pip install spacy
#!python -m spacy download en_core_web_sm

import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    # Normalize all types of single and double quotation marks to standard forms
    text = re.sub(r"[‘’`]", "'", text)  # Convert all single quote variations to '
    text = re.sub(r"[“”]", '"', text)   # Convert all double quote variations to "

    # Tokenize with regex to handle punctuation outside of words and contractions
    tokens = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text, re.UNICODE)

    text = ' '.join(tokens)
    # Process the text with spaCy's NLP pipeline
    doc = nlp(text)
    # Extract and join lemmas for each token
    lemmatized_text = ' '.join(token.lemma_ for token in doc)
    return lemmatized_text

# test the lemmatisation on simple and extreme cases

text = "How are you? I'm well thank you. doesn't. cleaning busiest"
lemmatize_text(text)

"""Ok now that we have a lemmatization function that works even on the extrem cases lets clean the text"""

def clean_data_lemma(df):
  df_copy = df.copy()
  df_copy["text"] = df_copy["text"].astype(str).apply(lemmatize_text)
  return df_copy

# test with df_aj and print the frist 5 text to test

data_aj_lemma = clean_data_lemma(df_aj)

for i in range(5):
  print(f"{i + 1}:\n{data_aj_lemma['text'][i]}")
  print(f"{df_aj['text'][i]}\n\n")

# now clean the rest of the df
df_bbc_lemma = clean_data_lemma(df_bbc)
df_jp_lemma = clean_data_lemma(df_jp)
df_nyt_lemma = clean_data_lemma(df_nyt)

# store the all the cleaned data in files for further usage
# the name of the file should be <source>_word.csv with headr

df_aj_ponctuation.to_csv("A_J_word.csv", index=False)
df_bbc_ponctuation.to_csv("BBC_word.csv", index=False)
df_jp_ponctuation.to_csv("J_P_word.csv", index=False)
df_nyt_ponctuation.to_csv("NYT_word.csv", index=False)

df_aj_lemma = data_aj_lemma

df_aj_lemma.to_csv("A_J_lemma.csv", index=False)
df_bbc_lemma.to_csv("BBC_lemma.csv", index=False)
df_jp_lemma.to_csv("J_P_lemma.csv", index=False)
df_nyt_lemma.to_csv("NYT_lemma.csv", index=False)
