# Run some text through the model:
def translate_text(input_sentence, model, tokenizer, device):
    # Prepare the input text with task prefix
    input_text = "translate English to Shakespeare style English: " + input_sentence
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Generate the output sequence
    outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)

    # Decode and return the output text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text
