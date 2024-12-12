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
# sentence = "You are an idiot, would you please leave?" # Produces good
# sentence = "If I was going to tell you a story, what kind would you want to hear?" # Produces good
# sentence = "When should we meet to talk about the party? I think it is going to be quite difficult to plan."
#sentence = "Have you ever been to the north? I hear it is quite cold."
# sentence = "I must leave to find the duke, he owes me a lot of money." # Produces good
# sentence = "Have you seen my son James? He is a small boy about twelve years old."
# sentence = "Do you have no honor? I could kill you for that insult!"
# sentence = "I have never seen a dragon. What do they look like?"

# Interactive translation loop
while True:
    # Ask the user for input
    sentence = input("Please type a sentence to convert to Shakespeare (or press Enter to quit): \n")
    
    # Exit the loop if input is empty
    if not sentence.strip():
        print("Goodbye!")
        break
    
    # Translate the sentence
    translated_text = translate_text(sentence, model, tokenizer, device)
    
    # Print the result
    print("\nOriginal Text: " + sentence)
    print("Shakespearean Translation: " + translated_text)
    print()