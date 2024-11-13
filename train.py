# from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments


# dataset = [
#     {"modern": "Where are you?", "shakespeare": "Whither art thou?"},
#     {"modern": "I love you", "shakespeare": "I doth love thee"},
#     # Add more pairs here
# ]



# model = T5ForConditionalGeneration.from_pretrained("t5-small")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")


# input_text = "translate English to Shakespeare: Where are you?"
# target_text = "Whither art thou?"

# # Tokenize inputs
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids
# target_ids = tokenizer(target_text, return_tensors="pt").input_ids

# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     save_steps=10_000,
#     save_total_limit=2,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     tokenizer=tokenizer,
# )

# trainer.train()

# print("Hiya")



from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# the following 2 hyperparameters are task-specific
max_source_length = 512
max_target_length = 128

# Suppose we have the following 2 training examples:
input_sequence_1 = "Where are you?"
output_sequence_1 = "Whither art thou?"

input_sequence_2 = "Sickness kept him home the third week"
output_sequence_2 = "Sickness hath kept that gent home the third week"

# encode the inputs
task_prefix = "translate English to Early Modern English: "
input_sequences = [input_sequence_1, input_sequence_2]

encoding = tokenizer(
    [task_prefix + sequence for sequence in input_sequences],
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)

input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

# encode the targets
target_encoding = tokenizer(
    [output_sequence_1, output_sequence_2],
    padding="longest",
    max_length=max_target_length,
    truncation=True,
    return_tensors="pt",
)
labels = target_encoding.input_ids

# replace padding token id's of the labels by -100 so it's ignored by the loss
labels[labels == tokenizer.pad_token_id] = -100

# forward pass
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
print(loss.item())

#Test the output
input_ids = tokenizer("translate English to Early Modern English: What language do you speak?.", return_tensors="pt").input_ids

outputs = model.generate(input_ids)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))