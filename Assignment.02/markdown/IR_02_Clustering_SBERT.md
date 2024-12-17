<a href="https://colab.research.google.com/github/dattali18/IR_Assignments/blob/main/Assignment.02/notebooks/IR_02_Clustering_SBERT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Clustering

## 1. Objective

Take the document (meaning vector representation of the document, the ouput of the last assginment `Doc2Vec`, `BERT`, `Sentence-BERT`, times 4 group from each) and cluster them into groups and compare the results with the actuale division form each publication.

## 2. Input

4 Groups of matrices each line repesent a document in it's vector form, from:

1. `Doc2Vec`
2. `BERT`
3. `Sentence-BERT`

### 3. Task

- Combine the four matrices into a single matrix for each technique.
- Apply clustering using:
- **K-Means** (with `k=4` for 4 journals).
- **DBSCAN** (select `eps` and `min_samples` heuristically).
- **Gaussian Mixture Model**.
- Evaluate the clusters using:
- Metrics: Precision, Recall, F1-Score, Accuracy.
- Visualization: Use UMAP, t-SNE, or other tools (e.g., Seaborn).


### 4. Output

- The plot of the real clusters vs. the clusters from the 3 methods mentioned above.
- The metrics for each clustering method.

# Doc2Vec Matrices

We have 4 `.csv` files with each $(100, \approx 600)$ and we need to combine them into one big matrix and then cluster them.

## Plan

1. Download the files from my `GitHub`.
2. Add a `'cluster'` column for each file (`=0` for AJ etc...).
3. Cluster with `Kmeans` for `k=4`.
4. Write a function to find the right parameters for 4 clusters for `DBSCAN` (i.e. the `eps` and `min_samples` parameters).
5. Cluster with `DBSCAN`
6. Cluster with `GMM`
7. Use `t-SNE` to visualize the cluster in $\mathbb{R}^2$
8. Output the plot for each clustering methods + original
9. Measure each method using the metric mentioned above.


```python
import warnings

warnings.filterwarnings("ignore")
```

Download the files from my `GitHub`


```python
sbert_link = "https://github.com/dattali18/IR_Assignments/blob/main/Assignment.02/data/bert-sbert/sbert_vectors.csv?raw=true"
```


```python
import pandas as pd

# download the data using the link
df_original = pd.read_csv(sbert_link)
df_original.head()
```





<div id="df-8ccf5d30-0c10-451a-ac58-0b1ee8f1e159" class="colab-df-container">
<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
	vertical-align: middle;
}

.dataframe tbody tr th {
	vertical-align: top;
}

.dataframe thead th {
	text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>Sheet</th>
  <th>RowIndex</th>
  <th>Dim0</th>
  <th>Dim1</th>
  <th>Dim2</th>
  <th>Dim3</th>
  <th>Dim4</th>
  <th>Dim5</th>
  <th>Dim6</th>
  <th>Dim7</th>
  <th>...</th>
  <th>Dim374</th>
  <th>Dim375</th>
  <th>Dim376</th>
  <th>Dim377</th>
  <th>Dim378</th>
  <th>Dim379</th>
  <th>Dim380</th>
  <th>Dim381</th>
  <th>Dim382</th>
  <th>Dim383</th>
</tr>
</thead>
<tbody>
<tr>
  <th>0</th>
  <td>A-J</td>
  <td>0</td>
  <td>0.004812</td>
  <td>0.026923</td>
  <td>0.015163</td>
  <td>0.007640</td>
  <td>-0.012498</td>
  <td>0.010280</td>
  <td>0.012724</td>
  <td>-0.018092</td>
  <td>...</td>
  <td>0.008701</td>
  <td>0.016885</td>
  <td>0.008800</td>
  <td>0.008313</td>
  <td>0.008201</td>
  <td>0.011964</td>
  <td>0.007581</td>
  <td>-0.013848</td>
  <td>-0.011848</td>
  <td>-0.001732</td>
</tr>
<tr>
  <th>1</th>
  <td>A-J</td>
  <td>1</td>
  <td>0.014052</td>
  <td>0.029228</td>
  <td>0.007936</td>
  <td>-0.016101</td>
  <td>-0.013189</td>
  <td>-0.001558</td>
  <td>-0.003385</td>
  <td>-0.022059</td>
  <td>...</td>
  <td>0.021585</td>
  <td>-0.000603</td>
  <td>-0.018857</td>
  <td>-0.014222</td>
  <td>-0.011867</td>
  <td>0.018346</td>
  <td>0.023300</td>
  <td>-0.006670</td>
  <td>-0.005978</td>
  <td>-0.006608</td>
</tr>
<tr>
  <th>2</th>
  <td>A-J</td>
  <td>2</td>
  <td>0.022085</td>
  <td>0.010386</td>
  <td>0.011578</td>
  <td>-0.003908</td>
  <td>0.006717</td>
  <td>-0.004299</td>
  <td>0.003057</td>
  <td>-0.019972</td>
  <td>...</td>
  <td>0.012670</td>
  <td>-0.010248</td>
  <td>-0.031286</td>
  <td>-0.005751</td>
  <td>0.011272</td>
  <td>0.025057</td>
  <td>0.023119</td>
  <td>-0.002533</td>
  <td>-0.021188</td>
  <td>0.009121</td>
</tr>
<tr>
  <th>3</th>
  <td>A-J</td>
  <td>3</td>
  <td>0.014390</td>
  <td>0.033544</td>
  <td>0.001297</td>
  <td>-0.012795</td>
  <td>0.030116</td>
  <td>-0.001583</td>
  <td>0.001665</td>
  <td>-0.034225</td>
  <td>...</td>
  <td>0.000973</td>
  <td>0.007662</td>
  <td>-0.006293</td>
  <td>0.002391</td>
  <td>-0.021095</td>
  <td>0.000995</td>
  <td>0.030528</td>
  <td>-0.000011</td>
  <td>-0.000507</td>
  <td>-0.014717</td>
</tr>
<tr>
  <th>4</th>
  <td>A-J</td>
  <td>4</td>
  <td>0.008542</td>
  <td>0.048237</td>
  <td>-0.007828</td>
  <td>-0.007906</td>
  <td>-0.001495</td>
  <td>-0.015571</td>
  <td>-0.003299</td>
  <td>-0.019079</td>
  <td>...</td>
  <td>0.004528</td>
  <td>0.000808</td>
  <td>-0.002086</td>
  <td>-0.010071</td>
  <td>0.007890</td>
  <td>0.029334</td>
  <td>0.019566</td>
  <td>0.002759</td>
  <td>0.001877</td>
  <td>-0.011802</td>
</tr>
</tbody>
</table>
<p>5 rows × 386 columns</p>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-8ccf5d30-0c10-451a-ac58-0b1ee8f1e159')"
		title="Convert this dataframe to an interactive table."
		style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
</svg>
</button>

<style>
.colab-df-container {
  display:flex;
  gap: 12px;
}

.colab-df-convert {
  background-color: #E8F0FE;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  display: none;
  fill: #1967D2;
  height: 32px;
  padding: 0 0 0 0;
  width: 32px;
}

.colab-df-convert:hover {
  background-color: #E2EBFA;
  box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
  fill: #174EA6;
}

.colab-df-buttons div {
  margin-bottom: 4px;
}

[theme=dark] .colab-df-convert {
  background-color: #3B4455;
  fill: #D2E3FC;
}

[theme=dark] .colab-df-convert:hover {
  background-color: #434B5C;
  box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
  filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
  fill: #FFFFFF;
}
</style>

<script>
  const buttonEl =
	document.querySelector('#df-8ccf5d30-0c10-451a-ac58-0b1ee8f1e159 button.colab-df-convert');
  buttonEl.style.display =
	google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
	const element = document.querySelector('#df-8ccf5d30-0c10-451a-ac58-0b1ee8f1e159');
	const dataTable =
	  await google.colab.kernel.invokeFunction('convertToInteractive',
												[key], {});
	if (!dataTable) return;

	const docLinkHtml = 'Like what you see? Visit the ' +
	  '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
	  + ' to learn more about interactive tables.';
	element.innerHTML = '';
	dataTable['output_type'] = 'display_data';
	await google.colab.output.renderOutput(dataTable, element);
	const docLink = document.createElement('div');
	docLink.innerHTML = docLinkHtml;
	element.appendChild(docLink);
  }
</script>
</div>


<div id="df-5a3639d1-617c-4146-ba58-0b45e1f9f70a">
<button class="colab-df-quickchart" onclick="quickchart('df-5a3639d1-617c-4146-ba58-0b45e1f9f70a')"
		title="Suggest charts"
		style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
 width="24px">
<g>
	<path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
</g>
</svg>
</button>

<style>
.colab-df-quickchart {
  --bg-color: #E8F0FE;
  --fill-color: #1967D2;
  --hover-bg-color: #E2EBFA;
  --hover-fill-color: #174EA6;
  --disabled-fill-color: #AAA;
  --disabled-bg-color: #DDD;
}

[theme=dark] .colab-df-quickchart {
  --bg-color: #3B4455;
  --fill-color: #D2E3FC;
  --hover-bg-color: #434B5C;
  --hover-fill-color: #FFFFFF;
  --disabled-bg-color: #3B4455;
  --disabled-fill-color: #666;
}

.colab-df-quickchart {
background-color: var(--bg-color);
border: none;
border-radius: 50%;
cursor: pointer;
display: none;
fill: var(--fill-color);
height: 32px;
padding: 0;
width: 32px;
}

.colab-df-quickchart:hover {
background-color: var(--hover-bg-color);
box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
fill: var(--button-hover-fill-color);
}

.colab-df-quickchart-complete:disabled,
.colab-df-quickchart-complete:disabled:hover {
background-color: var(--disabled-bg-color);
fill: var(--disabled-fill-color);
box-shadow: none;
}

.colab-df-spinner {
border: 2px solid var(--fill-color);
border-color: transparent;
border-bottom-color: var(--fill-color);
animation:
  spin 1s steps(1) infinite;
}

@keyframes spin {
0% {
  border-color: transparent;
  border-bottom-color: var(--fill-color);
  border-left-color: var(--fill-color);
}
20% {
  border-color: transparent;
  border-left-color: var(--fill-color);
  border-top-color: var(--fill-color);
}
30% {
  border-color: transparent;
  border-left-color: var(--fill-color);
  border-top-color: var(--fill-color);
  border-right-color: var(--fill-color);
}
40% {
  border-color: transparent;
  border-right-color: var(--fill-color);
  border-top-color: var(--fill-color);
}
60% {
  border-color: transparent;
  border-right-color: var(--fill-color);
}
80% {
  border-color: transparent;
  border-right-color: var(--fill-color);
  border-bottom-color: var(--fill-color);
}
90% {
  border-color: transparent;
  border-bottom-color: var(--fill-color);
}
}
</style>

<script>
async function quickchart(key) {
  const quickchartButtonEl =
	document.querySelector('#' + key + ' button');
  quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
  quickchartButtonEl.classList.add('colab-df-spinner');
  try {
	const charts = await google.colab.kernel.invokeFunction(
		'suggestCharts', [key], {});
  } catch (error) {
	console.error('Error during call to suggestCharts:', error);
  }
  quickchartButtonEl.classList.remove('colab-df-spinner');
  quickchartButtonEl.classList.add('colab-df-quickchart-complete');
}
(() => {
  let quickchartButtonEl =
	document.querySelector('#df-5a3639d1-617c-4146-ba58-0b45e1f9f70a button');
  quickchartButtonEl.style.display =
	google.colab.kernel.accessAllowed ? 'block' : 'none';
})();
</script>
</div>

</div>
</div>





```python
# create a new df where whe have the following column
# 1. Sheet with the name 'A-J', 'BBC', 'J-P', 'NY-T'
# 2. RowIndex - int
# 3. vector the numpy vector represntation (get all the value from Dim0 - Dim383 and make a list from them)

df = pd.DataFrame()

df['Sheet'] = df_original['Sheet']

df['vector'] = df_original.iloc[:, 2:].values.tolist()

df.head()
```





<div id="df-78fac6ca-10dd-4429-8dfb-1dcbdbe9d1ff" class="colab-df-container">
<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
	vertical-align: middle;
}

.dataframe tbody tr th {
	vertical-align: top;
}

.dataframe thead th {
	text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>Sheet</th>
  <th>vector</th>
</tr>
</thead>
<tbody>
<tr>
  <th>0</th>
  <td>A-J</td>
  <td>[0.0048119747079908, 0.0269228070974349, 0.015...</td>
</tr>
<tr>
  <th>1</th>
  <td>A-J</td>
  <td>[0.014051909558475, 0.0292279813438653, 0.0079...</td>
</tr>
<tr>
  <th>2</th>
  <td>A-J</td>
  <td>[0.0220847800374031, 0.0103860776871442, 0.011...</td>
</tr>
<tr>
  <th>3</th>
  <td>A-J</td>
  <td>[0.0143899163231253, 0.0335444100201129, 0.001...</td>
</tr>
<tr>
  <th>4</th>
  <td>A-J</td>
  <td>[0.0085417730733752, 0.0482366234064102, -0.00...</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-78fac6ca-10dd-4429-8dfb-1dcbdbe9d1ff')"
		title="Convert this dataframe to an interactive table."
		style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
</svg>
</button>

<style>
.colab-df-container {
  display:flex;
  gap: 12px;
}

.colab-df-convert {
  background-color: #E8F0FE;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  display: none;
  fill: #1967D2;
  height: 32px;
  padding: 0 0 0 0;
  width: 32px;
}

.colab-df-convert:hover {
  background-color: #E2EBFA;
  box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
  fill: #174EA6;
}

.colab-df-buttons div {
  margin-bottom: 4px;
}

[theme=dark] .colab-df-convert {
  background-color: #3B4455;
  fill: #D2E3FC;
}

[theme=dark] .colab-df-convert:hover {
  background-color: #434B5C;
  box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
  filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
  fill: #FFFFFF;
}
</style>

<script>
  const buttonEl =
	document.querySelector('#df-78fac6ca-10dd-4429-8dfb-1dcbdbe9d1ff button.colab-df-convert');
  buttonEl.style.display =
	google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
	const element = document.querySelector('#df-78fac6ca-10dd-4429-8dfb-1dcbdbe9d1ff');
	const dataTable =
	  await google.colab.kernel.invokeFunction('convertToInteractive',
												[key], {});
	if (!dataTable) return;

	const docLinkHtml = 'Like what you see? Visit the ' +
	  '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
	  + ' to learn more about interactive tables.';
	element.innerHTML = '';
	dataTable['output_type'] = 'display_data';
	await google.colab.output.renderOutput(dataTable, element);
	const docLink = document.createElement('div');
	docLink.innerHTML = docLinkHtml;
	element.appendChild(docLink);
  }
</script>
</div>


<div id="df-d825bcf4-2ba1-4ac4-8efa-7c1a9ff41463">
<button class="colab-df-quickchart" onclick="quickchart('df-d825bcf4-2ba1-4ac4-8efa-7c1a9ff41463')"
		title="Suggest charts"
		style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
 width="24px">
<g>
	<path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
</g>
</svg>
</button>

<style>
.colab-df-quickchart {
  --bg-color: #E8F0FE;
  --fill-color: #1967D2;
  --hover-bg-color: #E2EBFA;
  --hover-fill-color: #174EA6;
  --disabled-fill-color: #AAA;
  --disabled-bg-color: #DDD;
}

[theme=dark] .colab-df-quickchart {
  --bg-color: #3B4455;
  --fill-color: #D2E3FC;
  --hover-bg-color: #434B5C;
  --hover-fill-color: #FFFFFF;
  --disabled-bg-color: #3B4455;
  --disabled-fill-color: #666;
}

.colab-df-quickchart {
background-color: var(--bg-color);
border: none;
border-radius: 50%;
cursor: pointer;
display: none;
fill: var(--fill-color);
height: 32px;
padding: 0;
width: 32px;
}

.colab-df-quickchart:hover {
background-color: var(--hover-bg-color);
box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
fill: var(--button-hover-fill-color);
}

.colab-df-quickchart-complete:disabled,
.colab-df-quickchart-complete:disabled:hover {
background-color: var(--disabled-bg-color);
fill: var(--disabled-fill-color);
box-shadow: none;
}

.colab-df-spinner {
border: 2px solid var(--fill-color);
border-color: transparent;
border-bottom-color: var(--fill-color);
animation:
  spin 1s steps(1) infinite;
}

@keyframes spin {
0% {
  border-color: transparent;
  border-bottom-color: var(--fill-color);
  border-left-color: var(--fill-color);
}
20% {
  border-color: transparent;
  border-left-color: var(--fill-color);
  border-top-color: var(--fill-color);
}
30% {
  border-color: transparent;
  border-left-color: var(--fill-color);
  border-top-color: var(--fill-color);
  border-right-color: var(--fill-color);
}
40% {
  border-color: transparent;
  border-right-color: var(--fill-color);
  border-top-color: var(--fill-color);
}
60% {
  border-color: transparent;
  border-right-color: var(--fill-color);
}
80% {
  border-color: transparent;
  border-right-color: var(--fill-color);
  border-bottom-color: var(--fill-color);
}
90% {
  border-color: transparent;
  border-bottom-color: var(--fill-color);
}
}
</style>

<script>
async function quickchart(key) {
  const quickchartButtonEl =
	document.querySelector('#' + key + ' button');
  quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
  quickchartButtonEl.classList.add('colab-df-spinner');
  try {
	const charts = await google.colab.kernel.invokeFunction(
		'suggestCharts', [key], {});
  } catch (error) {
	console.error('Error during call to suggestCharts:', error);
  }
  quickchartButtonEl.classList.remove('colab-df-spinner');
  quickchartButtonEl.classList.add('colab-df-quickchart-complete');
}
(() => {
  let quickchartButtonEl =
	document.querySelector('#df-d825bcf4-2ba1-4ac4-8efa-7c1a9ff41463 button');
  quickchartButtonEl.style.display =
	google.colab.kernel.accessAllowed ? 'block' : 'none';
})();
</script>
</div>

</div>
</div>





```python
df = df_new
```


```python
# standerdize the data mean=0 std=1

from sklearn.preprocessing import StandardScaler
import numpy as np


scaler = StandardScaler()

# apply to each line of the df

df['std_vector'] = df['vector'].apply(lambda x: scaler.fit_transform(np.array(x).reshape(-1, 1)).flatten())
df['std_vector'].head()
```




<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
	vertical-align: middle;
}

.dataframe tbody tr th {
	vertical-align: top;
}

.dataframe thead th {
	text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>std_vector</th>
</tr>
</thead>
<tbody>
<tr>
  <th>0</th>
  <td>[0.33769827578570766, 1.9186505569411698, 1.07...</td>
</tr>
<tr>
  <th>1</th>
  <td>[1.1357293086135947, 2.364315669413121, 0.6405...</td>
</tr>
<tr>
  <th>2</th>
  <td>[1.4762320509213087, 0.6868184955137784, 0.767...</td>
</tr>
<tr>
  <th>3</th>
  <td>[1.004912208125632, 2.3412263212831683, 0.0915...</td>
</tr>
<tr>
  <th>4</th>
  <td>[0.6223741096931129, 3.4501298118439143, -0.54...</td>
</tr>
</tbody>
</table>
</div><br><label><b>dtype:</b> object</label>




```python
cluster_map = {'A-J': 0, 'BBC': 1, 'J-P': 2, 'NY-T': 3}
df['cluster'] = df['Sheet'].map(cluster_map)
```


```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
```


```python
# visualize the real cluster using t-SNE

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0)

# transofrm the df['vector'] to dataframe with freatuer 0 - 299 for
df_copy = df['std_vector'].apply(pd.Series)

df_tsne = tsne.fit_transform(df_copy)

df_tsne = pd.DataFrame(df_tsne, columns=['x', 'y'])

df_tsne['cluster'] = df['cluster']
```


```python
# plot the df_tsne

reverse_cluster_map = {v: k for k, v in cluster_map.items()}

plt.figure(figsize=(10, 10))
# add labels
plt.title('Real Clustering')

# make color scheme red, blue, green etc


df_tsne['cluster'] = df_tsne['cluster'].map(reverse_cluster_map)

sns.scatterplot(data=df_tsne, x='x', y='y', hue='cluster')

plt.show()
```



![png](IR_02_Clustering_SBERT_files/IR_02_Clustering_SBERT_13_0.png)



### Kmeans


```python
kmeans = KMeans(n_clusters=4, random_state=0).fit(df['std_vector'].tolist())

df['cluster_kmeans'] = kmeans.labels_

df['cluster_kmeans'].head()
```




<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
	vertical-align: middle;
}

.dataframe tbody tr th {
	vertical-align: top;
}

.dataframe thead th {
	text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>cluster_kmeans</th>
</tr>
</thead>
<tbody>
<tr>
  <th>0</th>
  <td>0</td>
</tr>
<tr>
  <th>1</th>
  <td>2</td>
</tr>
<tr>
  <th>2</th>
  <td>0</td>
</tr>
<tr>
  <th>3</th>
  <td>1</td>
</tr>
<tr>
  <th>4</th>
  <td>3</td>
</tr>
</tbody>
</table>
</div><br><label><b>dtype:</b> int32</label>




```python
# visutalize the cluster using the t-SNE df

df_tsne['cluster_kmeans'] = df['cluster_kmeans']

reverse_cluster_map = {v: k for k, v in cluster_map.items()}

plt.figure(figsize=(10, 10))

plt.title('Kmeans Clustering')

# make color scheme red, blue, green etc


df_tsne['cluster_kmeans'] = df_tsne['cluster_kmeans'].map(str)

sns.scatterplot(data=df_tsne, x='x', y='y', hue='cluster_kmeans')

plt.show()
```



![png](IR_02_Clustering_SBERT_files/IR_02_Clustering_SBERT_16_0.png)



### DBSCAN


```python
!pip install kneed
```

Collecting kneed
  Downloading kneed-0.8.5-py3-none-any.whl.metadata (5.5 kB)
Requirement already satisfied: numpy>=1.14.2 in /usr/local/lib/python3.10/dist-packages (from kneed) (1.26.4)
Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from kneed) (1.13.1)
Downloading kneed-0.8.5-py3-none-any.whl (10 kB)
Installing collected packages: kneed
Successfully installed kneed-0.8.5



```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.cluster import DBSCAN

def get_parameters(df, num_clusters=4, eps_adjustment=1.0, min_samples_adjustment=1):
X = np.array(df)

# Use NearestNeighbors to find the nearest neighbors
neighbors = NearestNeighbors(n_neighbors=2 * X.shape[1] - 1)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]

# Use KneeLocator to find the "elbow" point in the k-distance graph
kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve="convex", direction="increasing")
eps = distances[kneedle.elbow] * eps_adjustment

# Set min_samples to 2 * dimensions, another common heuristic
min_samples = 2 * X.shape[1] * min_samples_adjustment

return eps, min_samples

def find_best_parameters(df, num_clusters=4):
best_eps = None
best_min_samples = None
best_num_clusters = 0

for eps_adjustment in np.arange(0.5, 2.0, 0.1):
	for min_samples_adjustment in range(1, 5):
		eps, min_samples = get_parameters(df, num_clusters, eps_adjustment, min_samples_adjustment)
		db = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
		labels = db.labels_
		num_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

		if num_clusters_found == num_clusters:
			return eps, min_samples

		if num_clusters_found > best_num_clusters:
			best_eps = eps
			best_min_samples = min_samples
			best_num_clusters = num_clusters_found

return best_eps, best_min_samples

eps, min_samples = find_best_parameters(df['std_vector'].tolist(), num_clusters=4)
print(f"Best eps: {eps}, Best min_samples: {min_samples}")
```

Best eps: 17.583621196825618, Best min_samples: 768



```python
# use DBSCAN

dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df['std_vector'].tolist())

df['cluster_dbscan'] = dbscan.labels_
```


```python
# count the number of cluster

num_clusters = len(df['cluster_dbscan'].unique())

print(f"Number of clusters: {num_clusters}")
```

Number of clusters: 2



```python
# visualize data

# visutalize the cluster using the t-SNE df

df_tsne['cluster_dbscan'] = df['cluster_dbscan']

reverse_cluster_map = {v: k for k, v in cluster_map.items()}

plt.figure(figsize=(10, 10))

plt.title('DBSCAN Clustering')

# make color scheme red, blue, green etc


df_tsne['cluster_dbscan'] = df_tsne['cluster_dbscan'].map(str)

sns.scatterplot(data=df_tsne, x='x', y='y', hue='cluster_dbscan')

plt.show()
```



![png](IR_02_Clustering_SBERT_files/IR_02_Clustering_SBERT_22_0.png)



### GMM


```python
# apply GMM

gmm = GaussianMixture(n_components=4, random_state=0).fit(df['std_vector'].tolist())

df['cluster_gmm'] = gmm.predict(df['std_vector'].tolist())

df['cluster_gmm'].head()

```




<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
	vertical-align: middle;
}

.dataframe tbody tr th {
	vertical-align: top;
}

.dataframe thead th {
	text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>cluster_gmm</th>
</tr>
</thead>
<tbody>
<tr>
  <th>0</th>
  <td>0</td>
</tr>
<tr>
  <th>1</th>
  <td>2</td>
</tr>
<tr>
  <th>2</th>
  <td>0</td>
</tr>
<tr>
  <th>3</th>
  <td>1</td>
</tr>
<tr>
  <th>4</th>
  <td>3</td>
</tr>
</tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
# visualize data

# visutalize the cluster using the t-SNE df

df_tsne['cluster_gmm'] = df['cluster_gmm']

reverse_cluster_map = {v: k for k, v in cluster_map.items()}

plt.figure(figsize=(10, 10))

plt.title('GMM Clustering')

# make color scheme red, blue, green etc


df_tsne['cluster_gmm'] = df_tsne['cluster_gmm'].map(str)

sns.scatterplot(data=df_tsne, x='x', y='y', hue='cluster_gmm')

plt.show()
```



![png](IR_02_Clustering_SBERT_files/IR_02_Clustering_SBERT_25_0.png)



## Measurements


```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evluate_model(real, pred):
precision = precision_score(real, pred, average='macro')
recall = recall_score(real, pred, average='macro')
f1 = f1_score(real, pred, average='macro')
accuracy = accuracy_score(real, pred)

return precision, recall, f1, accuracy
```


```python
kmeans_evalutation_df = pd.DataFrame(columns=['precision', 'recall', 'f1', 'accuracy'])

kmeans_evalutation_df.loc['kmeans'] = evluate_model(df['cluster'], df['cluster_kmeans'])

kmeans_evalutation_df
```





<div id="df-c12d5aab-a3ed-40eb-9af7-1801a4eec26c" class="colab-df-container">
<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
	vertical-align: middle;
}

.dataframe tbody tr th {
	vertical-align: top;
}

.dataframe thead th {
	text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>precision</th>
  <th>recall</th>
  <th>f1</th>
  <th>accuracy</th>
</tr>
</thead>
<tbody>
<tr>
  <th>kmeans</th>
  <td>0.262264</td>
  <td>0.27372</td>
  <td>0.266735</td>
  <td>0.276215</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-c12d5aab-a3ed-40eb-9af7-1801a4eec26c')"
		title="Convert this dataframe to an interactive table."
		style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
</svg>
</button>

<style>
.colab-df-container {
  display:flex;
  gap: 12px;
}

.colab-df-convert {
  background-color: #E8F0FE;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  display: none;
  fill: #1967D2;
  height: 32px;
  padding: 0 0 0 0;
  width: 32px;
}

.colab-df-convert:hover {
  background-color: #E2EBFA;
  box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
  fill: #174EA6;
}

.colab-df-buttons div {
  margin-bottom: 4px;
}

[theme=dark] .colab-df-convert {
  background-color: #3B4455;
  fill: #D2E3FC;
}

[theme=dark] .colab-df-convert:hover {
  background-color: #434B5C;
  box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
  filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
  fill: #FFFFFF;
}
</style>

<script>
  const buttonEl =
	document.querySelector('#df-c12d5aab-a3ed-40eb-9af7-1801a4eec26c button.colab-df-convert');
  buttonEl.style.display =
	google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
	const element = document.querySelector('#df-c12d5aab-a3ed-40eb-9af7-1801a4eec26c');
	const dataTable =
	  await google.colab.kernel.invokeFunction('convertToInteractive',
												[key], {});
	if (!dataTable) return;

	const docLinkHtml = 'Like what you see? Visit the ' +
	  '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
	  + ' to learn more about interactive tables.';
	element.innerHTML = '';
	dataTable['output_type'] = 'display_data';
	await google.colab.output.renderOutput(dataTable, element);
	const docLink = document.createElement('div');
	docLink.innerHTML = docLinkHtml;
	element.appendChild(docLink);
  }
</script>
</div>


<div id="id_f0e37341-18f4-4593-9e0c-27e67f09603d">
<style>
  .colab-df-generate {
	background-color: #E8F0FE;
	border: none;
	border-radius: 50%;
	cursor: pointer;
	display: none;
	fill: #1967D2;
	height: 32px;
	padding: 0 0 0 0;
	width: 32px;
  }

  .colab-df-generate:hover {
	background-color: #E2EBFA;
	box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
	fill: #174EA6;
  }

  [theme=dark] .colab-df-generate {
	background-color: #3B4455;
	fill: #D2E3FC;
  }

  [theme=dark] .colab-df-generate:hover {
	background-color: #434B5C;
	box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
	filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
	fill: #FFFFFF;
  }
</style>
<button class="colab-df-generate" onclick="generateWithVariable('kmeans_evalutation_df')"
		title="Generate code using this dataframe."
		style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
   width="24px">
<path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
</svg>
</button>
<script>
  (() => {
  const buttonEl =
	document.querySelector('#id_f0e37341-18f4-4593-9e0c-27e67f09603d button.colab-df-generate');
  buttonEl.style.display =
	google.colab.kernel.accessAllowed ? 'block' : 'none';

  buttonEl.onclick = () => {
	google.colab.notebook.generateWithVariable('kmeans_evalutation_df');
  }
  })();
</script>
</div>

</div>
</div>





```python
# same for DBSCAN but map -1 to 2
df['cluster_dbscan'] = df['cluster_dbscan'].map(lambda x: x if x != -1 else 2)

dbscan_evalutation_df = pd.DataFrame(columns=['precision', 'recall', 'f1', 'accuracy'])

dbscan_evalutation_df.loc['dbscan'] = evluate_model(df['cluster'], df['cluster_dbscan'])

dbscan_evalutation_df
```





<div id="df-82370786-2e99-4609-ba14-2e82c53479e0" class="colab-df-container">
<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
	vertical-align: middle;
}

.dataframe tbody tr th {
	vertical-align: top;
}

.dataframe thead th {
	text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>precision</th>
  <th>recall</th>
  <th>f1</th>
  <th>accuracy</th>
</tr>
</thead>
<tbody>
<tr>
  <th>dbscan</th>
  <td>0.116767</td>
  <td>0.241653</td>
  <td>0.148723</td>
  <td>0.246803</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-82370786-2e99-4609-ba14-2e82c53479e0')"
		title="Convert this dataframe to an interactive table."
		style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
</svg>
</button>

<style>
.colab-df-container {
  display:flex;
  gap: 12px;
}

.colab-df-convert {
  background-color: #E8F0FE;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  display: none;
  fill: #1967D2;
  height: 32px;
  padding: 0 0 0 0;
  width: 32px;
}

.colab-df-convert:hover {
  background-color: #E2EBFA;
  box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
  fill: #174EA6;
}

.colab-df-buttons div {
  margin-bottom: 4px;
}

[theme=dark] .colab-df-convert {
  background-color: #3B4455;
  fill: #D2E3FC;
}

[theme=dark] .colab-df-convert:hover {
  background-color: #434B5C;
  box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
  filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
  fill: #FFFFFF;
}
</style>

<script>
  const buttonEl =
	document.querySelector('#df-82370786-2e99-4609-ba14-2e82c53479e0 button.colab-df-convert');
  buttonEl.style.display =
	google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
	const element = document.querySelector('#df-82370786-2e99-4609-ba14-2e82c53479e0');
	const dataTable =
	  await google.colab.kernel.invokeFunction('convertToInteractive',
												[key], {});
	if (!dataTable) return;

	const docLinkHtml = 'Like what you see? Visit the ' +
	  '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
	  + ' to learn more about interactive tables.';
	element.innerHTML = '';
	dataTable['output_type'] = 'display_data';
	await google.colab.output.renderOutput(dataTable, element);
	const docLink = document.createElement('div');
	docLink.innerHTML = docLinkHtml;
	element.appendChild(docLink);
  }
</script>
</div>


<div id="id_dc50fd6c-2a2b-40a6-b94e-57f06c85d8cc">
<style>
  .colab-df-generate {
	background-color: #E8F0FE;
	border: none;
	border-radius: 50%;
	cursor: pointer;
	display: none;
	fill: #1967D2;
	height: 32px;
	padding: 0 0 0 0;
	width: 32px;
  }

  .colab-df-generate:hover {
	background-color: #E2EBFA;
	box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
	fill: #174EA6;
  }

  [theme=dark] .colab-df-generate {
	background-color: #3B4455;
	fill: #D2E3FC;
  }

  [theme=dark] .colab-df-generate:hover {
	background-color: #434B5C;
	box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
	filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
	fill: #FFFFFF;
  }
</style>
<button class="colab-df-generate" onclick="generateWithVariable('dbscan_evalutation_df')"
		title="Generate code using this dataframe."
		style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
   width="24px">
<path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
</svg>
</button>
<script>
  (() => {
  const buttonEl =
	document.querySelector('#id_dc50fd6c-2a2b-40a6-b94e-57f06c85d8cc button.colab-df-generate');
  buttonEl.style.display =
	google.colab.kernel.accessAllowed ? 'block' : 'none';

  buttonEl.onclick = () => {
	google.colab.notebook.generateWithVariable('dbscan_evalutation_df');
  }
  })();
</script>
</div>

</div>
</div>





```python
# same for GMM

gmm_evalutation_df = pd.DataFrame(columns=['precision', 'recall', 'f1', 'accuracy'])

gmm_evalutation_df.loc['gmm'] = evluate_model(df['cluster'], df['cluster_gmm'])

gmm_evalutation_df
```





<div id="df-448576ec-17d0-41c9-ae5a-883240abdc8f" class="colab-df-container">
<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
	vertical-align: middle;
}

.dataframe tbody tr th {
	vertical-align: top;
}

.dataframe thead th {
	text-align: right;
}
</style>
<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
  <th></th>
  <th>precision</th>
  <th>recall</th>
  <th>f1</th>
  <th>accuracy</th>
</tr>
</thead>
<tbody>
<tr>
  <th>gmm</th>
  <td>0.262232</td>
  <td>0.27372</td>
  <td>0.266698</td>
  <td>0.276215</td>
</tr>
</tbody>
</table>
</div>
<div class="colab-df-buttons">

<div class="colab-df-container">
<button class="colab-df-convert" onclick="convertToInteractive('df-448576ec-17d0-41c9-ae5a-883240abdc8f')"
		title="Convert this dataframe to an interactive table."
		style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
<path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
</svg>
</button>

<style>
.colab-df-container {
  display:flex;
  gap: 12px;
}

.colab-df-convert {
  background-color: #E8F0FE;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  display: none;
  fill: #1967D2;
  height: 32px;
  padding: 0 0 0 0;
  width: 32px;
}

.colab-df-convert:hover {
  background-color: #E2EBFA;
  box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
  fill: #174EA6;
}

.colab-df-buttons div {
  margin-bottom: 4px;
}

[theme=dark] .colab-df-convert {
  background-color: #3B4455;
  fill: #D2E3FC;
}

[theme=dark] .colab-df-convert:hover {
  background-color: #434B5C;
  box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
  filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
  fill: #FFFFFF;
}
</style>

<script>
  const buttonEl =
	document.querySelector('#df-448576ec-17d0-41c9-ae5a-883240abdc8f button.colab-df-convert');
  buttonEl.style.display =
	google.colab.kernel.accessAllowed ? 'block' : 'none';

  async function convertToInteractive(key) {
	const element = document.querySelector('#df-448576ec-17d0-41c9-ae5a-883240abdc8f');
	const dataTable =
	  await google.colab.kernel.invokeFunction('convertToInteractive',
												[key], {});
	if (!dataTable) return;

	const docLinkHtml = 'Like what you see? Visit the ' +
	  '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
	  + ' to learn more about interactive tables.';
	element.innerHTML = '';
	dataTable['output_type'] = 'display_data';
	await google.colab.output.renderOutput(dataTable, element);
	const docLink = document.createElement('div');
	docLink.innerHTML = docLinkHtml;
	element.appendChild(docLink);
  }
</script>
</div>


<div id="id_4fcf0048-14eb-44d8-9511-0493b34d31b1">
<style>
  .colab-df-generate {
	background-color: #E8F0FE;
	border: none;
	border-radius: 50%;
	cursor: pointer;
	display: none;
	fill: #1967D2;
	height: 32px;
	padding: 0 0 0 0;
	width: 32px;
  }

  .colab-df-generate:hover {
	background-color: #E2EBFA;
	box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
	fill: #174EA6;
  }

  [theme=dark] .colab-df-generate {
	background-color: #3B4455;
	fill: #D2E3FC;
  }

  [theme=dark] .colab-df-generate:hover {
	background-color: #434B5C;
	box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
	filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
	fill: #FFFFFF;
  }
</style>
<button class="colab-df-generate" onclick="generateWithVariable('gmm_evalutation_df')"
		title="Generate code using this dataframe."
		style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
   width="24px">
<path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
</svg>
</button>
<script>
  (() => {
  const buttonEl =
	document.querySelector('#id_4fcf0048-14eb-44d8-9511-0493b34d31b1 button.colab-df-generate');
  buttonEl.style.display =
	google.colab.kernel.accessAllowed ? 'block' : 'none';

  buttonEl.onclick = () => {
	google.colab.notebook.generateWithVariable('gmm_evalutation_df');
  }
  })();
</script>
</div>

</div>
</div>



