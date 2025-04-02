import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model
MODEL_PATH = "C:/Users/emon1/Desktop/Practice Problem/eid/t5-small-summarizer-lora"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Streamlit UI
st.title("AI-Powered Text Summarization with LoRA Fine-Tuning")

# User input
user_text = st.text_area("Enter text to summarize", height=200)

if st.button("Summarize"):
    if user_text.strip():
        # Tokenize the user input
        inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        
        # Generate summary
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=50, num_beams=4)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display the summary
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text.")
