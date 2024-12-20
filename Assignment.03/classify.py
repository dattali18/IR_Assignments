import pandas as pd
import re

import requests

# Load keywords
with open('israel.txt') as f:
    pro_israeli_words = [line.strip().lower() for line in f]

with open('palestine.txt') as f:
    pro_palestinian_words = [line.strip().lower() for line in f]

# Define a function to match sentences
def extract_relevant_sentences(df, pro_israeli_words, pro_palestinian_words):
    extracted = []

    for idx, row in df.iterrows():
        doc_id = row['id']
        document = row['document']

        # Split into sentences
        sentences = re.split(r'[.!?]', document)  # Basic sentence splitting

        for sentence in sentences:
            sentence = sentence.strip().lower()
            is_pro_israeli = any(word in sentence for word in pro_israeli_words)
            is_pro_palestinian = any(word in sentence for word in pro_palestinian_words)

            if is_pro_israeli and not is_pro_palestinian:
                extracted.append((doc_id, sentence, 'pro-israeli'))
            elif is_pro_palestinian and not is_pro_israeli:
                extracted.append((doc_id, sentence, 'pro-palestinian'))

    return pd.DataFrame(extracted, columns=['id', 'sentence', 'type'])


def download_data(output_filename):
    url = "https://github.com/dattali18/IR_Assignments/blob/main/Assignment.01/data/data.xlsx?raw=true"

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_filename, "wb") as file:
            file.write(response.content)


def load_data(output_filename):
    data_aj = pd.read_excel(output_filename, sheet_name="A-J", engine="openpyxl")
    data_bbc = pd.read_excel(output_filename, sheet_name="BBC", engine="openpyxl")
    data_jp = pd.read_excel(output_filename, sheet_name="J-P", engine="openpyxl")
    data_nyt = pd.read_excel(output_filename, sheet_name="NY-T", engine="openpyxl")

    col_names_aj = ["title", "sub_title", "Body Text"]
    # we will add all the text from the 3 column above (is nan replace by "")
    # we will add a column 'id' that will be aj_<i> where i is the index of the row

    data_aj = data_aj[col_names_aj]
    data_aj = data_aj.fillna("")

    df_aj = pd.DataFrame()
    df_aj["id"] = range(1, len(data_aj) + 1)
    df_aj["id"] = "aj_" + df_aj["id"].astype(str)
    df_aj["document"] = (
        data_aj["title"] + " " + data_aj["sub_title"] + " " + data_aj["Body Text"]
    )

    # processing the data for bbc
    col_names_bbc = ["title", "Body Text"]

    data_bbc = data_bbc[col_names_bbc]
    data_bbc = data_bbc.fillna("")

    df_bbc = pd.DataFrame()
    df_bbc["id"] = range(1, len(data_bbc) + 1)
    df_bbc["id"] = "bbc_" + df_bbc["id"].astype(str)
    df_bbc["document"] = data_bbc["title"] + " " + data_bbc["Body Text"]

    # processing the data for nty
    col_names_nyt = ["title", "Body Text"]

    data_nyt = data_nyt[col_names_nyt]
    data_nyt = data_nyt.fillna("")

    df_nyt = pd.DataFrame()
    df_nyt["id"] = range(1, len(data_nyt) + 1)
    df_nyt["id"] = "nyt_" + df_nyt["id"].astype(str)
    df_nyt["document"] = data_nyt["title"] + " " + data_nyt["Body Text"]


def main():
    output_filename = "data/data.xlsx"

# # Apply to all DataFrames
# df_results = []
# for df in [df_source1, df_source2, df_source3, df_source4]:
#     df_results.append(extract_relevant_sentences(df, pro_israeli_words, pro_palestinian_words))

# # Combine results
# df_extracted = pd.concat(df_results)
# df_extracted.to_csv('extracted_sentences.csv', index=False)
