import pandas as pd
import re

import requests

pro_israel_words = [
    "self defense",
    "security",
    "sovereignty",
    "defend",
    "protect",
    "safety",
    "legitimate",
    "democratic",
    "righteous",
    "forces",
    "guards",
    "shield",
    "defense forces",
    "defense",
    "shield",
    "shelter",
    "resilient",
    "guard",
    "secure",
    "protect",
]

anti_israel_words = [
    "invasion",
    "assault",
    "siege",
    "raid",
    "destruction",
    "relentless",
    "excessive",
    "occupation",
    "violation",
    "aggressive",
    "military campaign",
    "blocking",
    "invasion",
    "strikes",
    "bombardment",
    "attack",
    "deadly",
    "force",
    "offensive",
    "raid",
]

pro_palestine_words = [
    "resistance",
    "rights",
    "humanitarian",
    "peaceful",
    "legitimate",
    "protesters",
    "activists",
    "demonstrations",
    "supporters",
    "advocates",
    "solidarity",
    "protest",
    "defend",
    "freedom",
    "struggle",
    "liberation",
    "justice",
    "human rights",
    "steadfast",
    "peaceful",
]

anti_palestine_words = [
    "militant",
    "rockets",
    "fired",
    "attack",
    "hostile",
    "aggressive",
    "launch",
    "threat",
    "violent",
    "extremist",
    "militant groups",
    "fighters",
    "attacks",
    "firing",
    "hostile",
    "aggressive",
    "radical",
    "threat",
    "extremist",
    "violent",
]


def extract_relevant_sentences(
    df,
    pro_israeli_words,
    pro_palestinian_words,
    anti_israeli_words,
    anti_palestinian_words,
):
    extracted = []

    for idx, row in df.iterrows():
        doc_id = row["id"]
        document = row["document"]

        sentences = re.split(r"[.!?]", document)

        for sentence in sentences:
            sentence = sentence.strip().lower()

            # Check for different sentiment combinations
            is_pro_israeli = any(word in sentence for word in pro_israeli_words)
            is_pro_palestinian = any(word in sentence for word in pro_palestinian_words)
            is_anti_israeli = any(word in sentence for word in anti_israeli_words)
            is_anti_palestinian = any(
                word in sentence for word in anti_palestinian_words
            )

            # Classification logic
            if is_pro_israeli and not (
                is_pro_palestinian or is_anti_israeli or is_anti_palestinian
            ):
                extracted.append((doc_id, sentence, "pro-israeli"))
            elif is_pro_palestinian and not (
                is_pro_israeli or is_anti_israeli or is_anti_palestinian
            ):
                extracted.append((doc_id, sentence, "pro-palestinian"))
            elif is_anti_israeli and not (
                is_pro_israeli or is_pro_palestinian or is_anti_palestinian
            ):
                extracted.append((doc_id, sentence, "anti-israeli"))
            elif is_anti_palestinian and not (
                is_pro_israeli or is_pro_palestinian or is_anti_israeli
            ):
                extracted.append((doc_id, sentence, "anti-palestinian"))
            elif not any(
                [
                    is_pro_israeli,
                    is_pro_palestinian,
                    is_anti_israeli,
                    is_anti_palestinian,
                ]
            ):
                extracted.append((doc_id, sentence, "neutral"))

    return pd.DataFrame(extracted, columns=["id", "sentence", "type"])


def download_data(output_filename):
    url = "https://github.com/dattali18/IR_Assignments/blob/main/Assignment.01/data/data.xlsx?raw=true"

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_filename, "wb") as file:
            file.write(response.content)


def load_data(output_filename):
    data_aj = pd.read_excel(output_filename, sheet_name="A-J")
    data_bbc = pd.read_excel(output_filename, sheet_name="BBC")
    data_jp = pd.read_excel(output_filename, sheet_name="J-P")
    data_nyt = pd.read_excel(output_filename, sheet_name="NY-T")

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

    col_names_jp = ["title", "Body"]

    data_jp = data_jp[col_names_jp]
    data_jp = data_jp.fillna("")

    df_jp = pd.DataFrame()
    df_jp["id"] = range(1, len(data_jp) + 1)
    df_jp["id"] = "jp_" + df_jp["id"].astype(str)
    df_jp["document"] = data_jp["Body"]

    return df_aj, df_bbc, df_nyt, df_jp


import re


def clean_text(text):
    # Normalize all types of single and double quotation marks to standard forms
    text = re.sub(r"[‘’`]", "'", text)  # Convert all single quote variations to '
    text = re.sub(r"[“”]", '"', text)  # Convert all double quote variations to "

    # remove any and all special characters since it will not be useful for our analysis
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    return text


def main():
    output_filename = "data/data.xlsx"

    # download_data(output_filename)
    df_aj, df_bbc, df_nyt, df_jp = load_data(output_filename)

    # Apply to all DataFrames
    df_results = []
    for df in [df_aj, df_bbc, df_nyt, df_jp]:
        df_results.append(
            extract_relevant_sentences(
                df,
                pro_israel_words,
                pro_palestine_words,
                anti_israel_words,
                anti_palestine_words,
            )
        )

    # remove special characters from the sentences
    df_results[0]["sentence"] = df_results[0]["sentence"].apply(clean_text)
    df_results[1]["sentence"] = df_results[1]["sentence"].apply(clean_text)
    df_results[2]["sentence"] = df_results[2]["sentence"].apply(clean_text)
    df_results[3]["sentence"] = df_results[3]["sentence"].apply(clean_text)

    # Combine results
    print("Combining results")
    df_extracted = pd.concat(df_results)
    df_extracted.to_csv("extracted_sentences.csv", index=False)

    print("Done")


if __name__ == "__main__":
    main()
