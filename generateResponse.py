# run_translation.py
from util.translate import translate_text
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("shakespeare_translation_tokenizer")
model = T5ForConditionalGeneration.from_pretrained("shakespeare_translation_model")

# Choose device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# Example sentence
sentence = "Have you been to the north, It is quite cold."

# Translate the sentence
translated_text = translate_text(sentence, model, tokenizer, device)

# Print the result
print("Shakespearean Translation:", translated_text)
