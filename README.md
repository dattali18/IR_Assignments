# IR_Assignments
This repo is the assignment in the Information Retrieval Course (IR).

## Assignment 1:

### Data Cleaning Script

This script is designed to preprocess and clean text data from an Excel file for further analysis. The script performs a series of data cleaning tasks, including text normalization, tokenization, and lemmatization. The cleaned data is then saved to CSV files.

#### File Structure

- **01_data_cleaning.py**: Main script for data cleaning and preprocessing.

#### Dependencies

The script requires the following Python libraries:
- `pandas`
- `os`
- `re`
- `spacy`
- `nltk`
- `typing`

Ensure you have these libraries installed before running the script.

#### Data Preprocessing

The script processes data from an Excel file located at `../data/data.xlsx`. It reads data from four sheets: "A-J", "BBC", "J-P", and "NY-T". Each sheet is processed separately, combining relevant columns into a single document column and adding an ID column.

##### Functions

- **process_aj(df_aj)**: Processes the "A-J" sheet.
- **process_bbc(df_bbc)**: Processes the "BBC" sheet.
- **process_jp(df_jp)**: Processes the "J-P" sheet.
- **process_nyt(df_nyt)**: Processes the "NY-T" sheet.
- **get_excel_data(path)**: Reads and processes the data from the Excel file.

#### Data Cleaning

The script includes functions to clean the text data by normalizing quotation marks, tokenizing text, and handling punctuation. It also performs lemmatization to standardize words to their base forms.

##### Functions

- **clean_text(text)**: Normalizes single and double quotation marks.
- **clean_df(df)**: Cleans the text in the DataFrame.
- **clean_word(text)**: Tokenizes text and handles punctuation.
- **clean_word_df(df)**: Cleans the words in the DataFrame.
- **clean_lemma(text)**: Performs lemmatization on the text.
- **clean_lemma_df(df)**: Cleans the lemmas in the DataFrame.

#### Saving Cleaned Data

The cleaned data is saved to CSV files in two folders: `../data/word` and `../data/lemma`.

##### Functions

- **save_documents_to_csv(df, file_path)**: Saves a DataFrame to a CSV file.

#### Main Function

The `main()` function orchestrates the data preprocessing and cleaning tasks. It reads the Excel data, cleans the text and words, performs lemmatization, and saves the cleaned data to CSV files.

#### Usage

To run the script, execute the following command:

```bash
python 01_data_cleaning.py
```

Ensure you have the necessary NLTK resources downloaded by running:

```bash
python -m spacy download en_core_web_sm
```

## Acknowledgements

This script is part of the assignments in the Information Retrieval Course (IR).
