# Assignment 02

## Assignment Overview: Clustering and Classification

### Part 1: Clustering

1. **Objective**:
    - Group the documents (articles) using clustering algorithms and evaluate the results.

2. **Input**:
    - Four document matrices per vectorization technique (Doc2Vec, BERT, Sentence-BERT), each with dimensions (100, 600).

3. **Tasks**:
    - Combine the four matrices into a single matrix for each technique.
    - Apply clustering using:
        - **K-Means** (with `k=4` for 4 journals).
        - **DBSCAN** (select `eps` and `min_samples` heuristically).
        - **Gaussian Mixture Model**.
    - Evaluate the clusters using:
        - Metrics: Precision, Recall, F1-Score, Accuracy.
        - Visualization: Use UMAP, t-SNE, or other tools (e.g., Seaborn).

### Part 2: Classification
1. **Objective**:
    - Build classifiers to predict the journal group.

2. **Algorithms**:
    - **Artificial Neural Network (ANN)** (two architectures provided):
        - ANN Architecture 1: RELU activation layers.
        - ANN Architecture 2: GELU activation layers.
    - **Other Classifiers**: Naive Bayes (NB), Support Vector Machine (SVM), Logistic Regression (LoR), Random Forest (RF).

3. **Tasks**:
    - Perform 10-fold cross-validation for all classifiers (except ANN).
    - Identify and rank the top 20 most important features for NB, RF, SVM, LoR.
    - Write explanations for feature importance in a README document and include the ranked lists in an Excel file.

4. **ANN Specifics**:
    - Split data: Train (80%, with 10% validation from the train set) and Test (20%).
    - Use the given ANN architectures with specific configurations:
        - Maximum 15 epochs.
        - Batch size: 32.
        - Early stopping after 3 validation iterations without improvement.
        - Save the best model (ModelCheckpoint).

### Deliverables

1. Python code files.
2. A detailed README with explanations, tables, plots, and insights.
3. Excel files with feature importance rankings.
4. All files zipped into a directory named after the student group.


