# Assignment 04 - Sentiment Analysis

We have divided the assignment into 3 steps:

1. Data Preparation (similar to what we did last time)
2. Model Training
3. Article Classification + Analysis

## Todo

- [x] Add more code example to the **Stage 1** and more explanation about our words choices
- [x] Do **Stage 2**
- [ ] Do **Stage 3**

## Stage 1 - Data Preparation

For preparing the data we have done something similar to last time (assignment 3) but needed to change in order to take the different requirements (5 class and not 3), so what we did is we wrote 4 different dictionary of words _pro-israel_, _pro-palestine_, _anti-israel_, _anti-palestine_

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

For the training of this model we used transfer learning from a pre-trained base **BERT** model.

The first step was to take a subset of 500 random instances from each class (to make the training easier) and then we used the `transformers` library to load the pre-trained model and tokenizer

We also spilt the data 85% - 15% for training and validation.

The following code is the `SentimentClassifier` class that we used to train the model

```python
# Create model class

class SentimentClassifier(nn.Module):
	def __init__(self, n_classes=5):
		super().__init__()
		self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
		self.drop = nn.Dropout(0.3)
		self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

	def forward(self, input_ids, attention_mask):
		output = self.bert(
		input_ids=input_ids,
		attention_mask=attention_mask
		)
		output = self.drop(output[0][:, 0, :])
		return self.fc(output)
```

The following code is the `train_model` function that we used to train the model

```python
# Training loop
def train_model():
	model.train()
	total_loss = 0
	for batch in tqdm(train_loader, desc='Training'):
		optimizer.zero_grad()
		input_ids = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		labels = batch['label'].to(device)
		outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	return total_loss / len(train_loader)
```

The following code is the `evaluate_model` function that we used to evaluate the model

```python
# Evaluation loop
def evaluate_model():
	model.eval()
	total_loss = 0
	all_predictions = []
	all_labels = []
	with torch.no_grad():
		for batch in tqdm(val_loader, desc='Evaluating'):
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['label'].to(device)
			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
			loss = criterion(outputs, labels)
			_, predictions = torch.max(outputs, dim=1)
			total_loss += loss.item()
			all_predictions.extend(predictions.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())
	accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
	return total_loss / len(val_loader), accuracy
```

The following code is the training loop

```python
# Training
for epoch in range(n_epochs):
	print(f'\nEpoch {epoch + 1}/{n_epochs}')
	train_loss = train_model()
	val_loss, val_accuracy = evaluate_model()
	print(f'Training Loss: {train_loss:.4f}')
	print(f'Validation Loss: {val_loss:.4f}')
	print(f'Validation Accuracy: {val_accuracy:.4f}')
```

The screenshot from the training process

![](training_1.png)

![](training_2.png)
As we can see after the final epoch the model has:

- Validation Loss: 0.1146
- Validation Accuracy: 0.9787

Meaning our model is accurate $97.87\%$ of the time.

## Stage 3.1 - Article Classification

## Stage 3.2 - Analysis
