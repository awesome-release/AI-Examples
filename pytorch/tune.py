from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import os

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("\n\n==== Loading dataset...")
if os.environ['TUNING_DATASET_LOCATION'].endswith(".json") or os.environ['TUNING_DATASET_LOCATION'].endswith(".jsonl"):
    dataset = load_dataset("json", data_files=os.environ['TUNING_DATASET_LOCATION'])
else:
    dataset = load_dataset(os.environ['TUNING_DATASET_LOCATION'])

tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_LOCATION_OR_NAME'])
tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=37).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=37).select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

print("\n\n==== Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(os.environ['MODEL_LOCATION_OR_NAME'])

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

print("\n\n==== Tuning model...")
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

print("\n\n==== Evaluating results...")
metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

print("\n\n==== Saving model...")

torch.save(model, "out")

print("\n\n==== Done!")
