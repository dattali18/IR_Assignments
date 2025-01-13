# Assignment 04 - Sentiment Analysis

We have divided the assignment into 3 steps:

1. Data Preparation (similar to what we did last time)
2. Model Training
3. Article Classification + Analysis

## Todo

- [ ] Add more code example to the **Stage 1** and more explanation about our words choices
- [ ] Do **Stage 2**
- [ ] Do **Stage 3**
## Stage 1 - Data Preparation

For preparing the data we have done something similar to last time (assignment 3) but needed to change in order to take the different requirements (5 class and not 3), so what we did is we wrote 4 different dictionary of words *pro-israel*, *pro-palestine*, *anti-israel*, *anti-palestine* 

```python
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
...
]

...
```

and we also changed the `extract_sentence` function from last time

```python
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
```

The idea is still the same as last time but with more logical ands and ors.

This gave us a first file with $40k$ sentence but the problem was the distribution of each class so we did an analysis to see the distribution and saw: (the file is `analysis.py`)

```plaintext
Sentiment Class Distribution:
------------------------------
neutral: 38421
pro-palestinian: 2186
pro-israeli: 2002
anti-israeli: 1657
anti-palestinian: 1222

Percentages:
------------------------------
neutral: 84.46%
pro-palestinian: 4.81%
pro-israeli: 4.40%
anti-israeli: 3.64%
anti-palestinian: 2.69%
```

So since there was so much of the `neutral` class I wrote a script to take a random subset from this class to balance out the distribution + added `int` labels to the data frame and got this

```plaintext
Balanced Dataset Distribution:
------------------------------
pro-palestinian: 2186
pro-israeli: 2002
anti-israeli: 1657
neutral: 1500
anti-palestinian: 1222

Percentages:
------------------------------
pro-palestinian: 25.52%
pro-israeli: 23.37%
anti-israeli: 19.34%
neutral: 17.51%
anti-palestinian: 14.26%
```

Which is better.
## Stage 2- Model Training

## Stage 3.1 - Article Classification

## Stage 3.2 - Analysis
