from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import torch
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from util.translate import translate_text
from util.checkpoint import load_checkpoint, save_checkpoint, get_latest_checkpoint



# Load model and tokenizer
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("shakespeare_translation_tokenizer")
model = T5ForConditionalGeneration.from_pretrained("shakespeare_translation_model")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
epochs = 95
batch_size = 8
learning_rate = 3e-5
max_source_length = 512
max_target_length = 128
sample_size = 20000  # number of random lines to sample
save_interval = 5
checkpoint_dir = "checkpoints"

# -------Prepare dataset-------
class ShakespeareDataset(Dataset):
    def __init__(self, modern_text, shakespeare_text, tokenizer, max_source_length, max_target_length):
    
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.modern_text = modern_text
        self.shakespeare_text = shakespeare_text
        
    def __len__(self):
        return len(self.modern_text)

    
    def __getitem__(self, idx):
        # Get the corresponding modern and Shakespearean lines
        # Prepare the inputs with task prefix
        task_prefix = "translate English to Shakspeare: "
        input_text = task_prefix + self.modern_text[idx]
        target_text = self.shakespeare_text[idx]

        # Tokenize inputs and outputs
        input_encoding = self.tokenizer(
            input_text, max_length=self.max_source_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text, max_length=self.max_target_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        # Return input IDs, attention mask, and labels
        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in the loss

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# -------Read files and select lines for training-------
# Read the file once to get the number of lines
with open('trainingdata/modern.txt', 'r') as f:
    total_lines = sum(1 for line in f)

# Generate a set of random line indices to sample from
sample_indices = set(random.sample(range(total_lines), sample_size))

# Open both files and read only the selected lines
modern_texts = []
shakespeare_texts = []

with open('trainingData/modern.txt', 'r') as modern_file, open('trainingData/original.txt', 'r') as shakespeare_file:
    for idx, (modern_line, shakespeare_line) in enumerate(zip(modern_file, shakespeare_file)):
        if idx in sample_indices:
            modern_texts.append(modern_line.strip())
            shakespeare_texts.append(shakespeare_line.strip())



# -------Create dataset and dataloader-------
# Split into train and validation sets
train_modern, val_modern, train_shakespeare, val_shakespeare = train_test_split(
    modern_texts, shakespeare_texts, test_size=0.2, random_state=42
)
train_dataset = ShakespeareDataset(train_modern, train_shakespeare, tokenizer, max_source_length, max_target_length)
val_dataset = ShakespeareDataset(val_modern, val_shakespeare, tokenizer, max_source_length, max_target_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Scheduler
max_lr = learning_rate  # maybe try 5e-5, for better performance
steps_per_epoch = len(train_dataloader)

scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs)

# Training loop
device = torch.device(device_type)
model.to(device)

# Load the checkpoint if it exists
latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    start_epoch = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
    print("Checkpoint found")
else:
    print("No checkpoint found, starting from scratch.")
    start_epoch = 0  # Start from the first epoch

for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(val_dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step the scheduler
        scheduler.step()

    # Logging the learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader):.4f}, Learning Rate: {current_lr}")
    

    # Save a checkpoint periodically
    if (epoch + 1) % save_interval == 0:
        save_checkpoint(model, optimizer, scheduler, epoch + 1, filepath=f"checkpoints/checkpoint_epoch_{epoch+1}.pth")

        # Validation step after training
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass for validation
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss:.4f}")



#-------Save Model-------
print("Saving Model")
model.save_pretrained("shakespeare_translation_model")
tokenizer.save_pretrained("shakespeare_translation_tokenizer")

testPhrase = "What kind of person are you? I think you are an idiot."

translated_text = translate_text(testPhrase, model, tokenizer, device)
print("Shakespearean Translation: " + translated_text)
