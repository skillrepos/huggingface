import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader

# Step 3: Load a small dataset
train_dataset = load_dataset('glue', 'sst2', split='train[:1%]')
test_dataset = load_dataset('glue', 'sst2', split='validation[:1%]')

# Step 4: Load tokenizer and model
model_name = 'distilbert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Step 5: Preprocess the data
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=128)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Step 6: Prepare data for training
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 7: Evaluate before fine-tuning
test_dataloader = DataLoader(tokenized_test, batch_size=16)

model.eval()

correct = 0
total = 0

for batch in test_dataloader:
    inputs = {'input_ids': batch['input_ids'],
              'attention_mask': batch['attention_mask']}
    labels = batch['label']
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

pre_fine_tune_accuracy = correct / total
print(f'Accuracy before fine-tuning: {pre_fine_tune_accuracy:.2f}')

# Step 8: Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    logging_steps=10,
    logging_dir='./logs',
    eval_strategy='no',
    save_strategy='no',
    disable_tqdm=True,
)

# Step 9: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
)

# Step 10: Fine-tune the model
trainer.train()

# Step 11: Evaluate after fine-tuning
model.eval()

correct = 0
total = 0

for batch in test_dataloader:
    inputs = {'input_ids': batch['input_ids'],
              'attention_mask': batch['attention_mask']}
    labels = batch['label']
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

post_fine_tune_accuracy = correct / total
print(f'Accuracy after fine-tuning: {post_fine_tune_accuracy:.2f}')

# Step 12: Display the difference
print(f'Accuracy before fine-tuning: {pre_fine_tune_accuracy:.2f}')
print(f'Accuracy after fine-tuning: {post_fine_tune_accuracy:.2f}')
