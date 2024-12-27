### Step 1: Install Necessary Libraries

```bash
pip install transformers pandas sklearn torch
```

### Step 2: Load and Preprocess Your Data

Load CSV file and preprocess the data:

```python
import pandas as pd

# Load the data
df = pd.read_csv('extracted_sentence.csv')

# Display the first few rows
print(df.head())

# Rename columns if necessary
df = df.rename(columns={'sentence': 'text', 'type': 'label'})

# Check for any missing values and handle them if necessary
df = df.dropna()

# Display the first few rows to confirm changes
print(df.head())

# Map labels to integers if they are not already
label_mapping = {'pro': 1, 'con': 0}  # Adjust this mapping if necessary
df['label'] = df['label'].map(label_mapping)
```

### Step 3: Split Your Data
Split your data into training and testing sets:

```python
from sklearn.model_selection import train_test_split

# Split the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Reset index to avoid any issues
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
```

### Step 4: Load a Pre-trained Model and Tokenizer
Choose a pre-trained model and tokenizer from Hugging Face:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'  # You can choose another model if you prefer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

### Step 5: Prepare the Data for the Model
Transform your text data into the format required by the model:

```python
import torch
from torch.utils.data import DataLoader, Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create Datasets
train_dataset = SentimentDataset(
    texts=train_df.text.to_numpy(),
    labels=train_df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=128
)

test_dataset = SentimentDataset(
    texts=test_df.text.to_numpy(),
    labels=test_df.label.to_numpy(),
    tokenizer=tokenizer,
    max_len=128
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
```

### Step 6: Train the Model
Define the training loop and train your model:

```python
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# Training parameters
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# Training loop
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_df)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')
```

### Step 7: Evaluate the Model
Evaluate your model on the test set:

```python
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

test_acc, test_loss = eval_model(
    model,
    test_loader,
    loss_fn,
    device,
    len(test_df)
)

print(f'Test loss {test_loss} accuracy {test_acc}')
```

### Step 8: Make Predictions on New Data
Finally, you can use your trained model to make predictions on new data:

```python
def predict_sentiment(text, model, tokenizer, device):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    _, prediction = torch.max(outputs.logits, dim=1)

    return prediction.item()

# Example prediction
new_text = "Your example text here"
sentiment = predict_sentiment(new_text, model, tokenizer, device)
print(f"Sentiment: {sentiment}")
```

This guide should help you get started with training sentiment analysis models using Hugging Face. If you have any specific questions or run into issues, feel free to ask!