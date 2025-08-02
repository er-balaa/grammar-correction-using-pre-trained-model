!pip install -q transformers sentencepiece

from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "prithivida/grammar_error_correcter_v1"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def correct_grammar(text):
    input_text = "gec: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        early_stopping=True,
        num_return_sequences=1
    )
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence

sentence = input("Enter a sentence: ")
print(f"\nOriginal: {sentence}")
print(f"Corrected: {correct_grammar(sentence)}")
