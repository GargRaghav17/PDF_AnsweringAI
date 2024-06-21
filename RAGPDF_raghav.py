# # # import fitz  # PyMuPDF
# # # from transformers import BertTokenizer, BertForQuestionAnswering
# # # import torch

# # # # Initialize the tokenizer and model
# # # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# # # model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# # # def extract_text_from_pdf(pdf_path):
# # #     document = fitz.open(pdf_path)
# # #     text = ""
# # #     for page_num in range(len(document)):
# # #         try:
# # #             page = document.load_page(page_num)
# # #             page_text = page.get_text()
# # #             if page_text:  # Check if the page has text
# # #                 text += page_text
# # #             else:
# # #                 print(f"Page {page_num + 1} has no extractable text.")
# # #         except Exception as e:
# # #             print(f"Error reading page {page_num + 1}: {e}")
# # #     return text

# # # def answer_question(question, text):
# # #     inputs = tokenizer(question, text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
# # #     with torch.no_grad():
# # #         outputs = model(**inputs)
# # #     answer_start = torch.argmax(outputs.start_logits)
# # #     answer_end = torch.argmax(outputs.end_logits) + 1
# # #     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
# # #     return answer

# # # if __name__ == '__main__':
# # #     # Specify the path to your PDF file
# # #     pdf_path = "c:/Users/hp/Desktop/engbookmerged.pdf"
    
# # #     # Extract text from the PDF
# # #     pdf_text = extract_text_from_pdf(pdf_path)
# # #     print("Extracted Text:")
# # #     print(pdf_text[:5000])  # Print the first 1000 characters to avoid too much output
    
# # #     # Ask a question about the PDF
# # #     question = "how did the birds fall?"
# # #     answer = answer_question(question, pdf_text)
    
# # #     print("\nQuestion:")
# # #     print(question)
# # #     print("\nAnswer:")
# # #     print(answer)

# # import streamlit as st
# # import os
# # import fitz  # PyMuPDF for PDF text extraction
# # from transformers import pipeline  # Hugging Face transformers for question answering
# # import nltk

# # # Download NLTK resources (needed for text processing)
# # nltk.download('punkt')

# # # Initialize the question-answering pipeline from transformers
# # qa_pipeline = pipeline("question-answering")

# # def extract_pdf_text(pdf_path):
# #     """Extract text from PDF file using PyMuPDF (fitz)."""
# #     text = ""
# #     doc = fitz.open(pdf_path)
# #     for page_num in range(len(doc)):
# #         text += doc[page_num].get_text()
# #     return text

# # def answer_question(question, context):
# #     """Answer a question based on context using transformers pipeline."""
# #     result = qa_pipeline(question=question, context=context)
# #     return result['answer']

# # def main():
# #     st.title("PDF Question Answering Chatbot")

# #     uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# #     if uploaded_file is not None:
# #         pdf_path = uploaded_file.name
# #         with open(pdf_path, "wb") as f:
# #             f.write(uploaded_file.getbuffer())

# #         pdf_text = extract_pdf_text(pdf_path)

# #         st.header("PDF Content")
# #         st.text(pdf_text[:500])  # Display first 500 characters of extracted text

# #         st.header("Ask a Question")
# #         question = st.text_input("Enter your question here")

# #         if st.button("Ask"):
# #             if question:
# #                 answer = answer_question(question, pdf_text)
# #                 st.success(f"Answer: {answer}")
# #             else:
# #                 st.warning("Please enter a question.")

# # if __name__ == "__main__":
# #     main()




# import streamlit as st
# import os
# import fitz  # PyMuPDF for PDF text extraction
# import torch
# from transformers import BertForQuestionAnswering, BertTokenizer
# import nltk

# # Download NLTK resources (needed for text processing)
# nltk.download('punkt')

# # Load the BERT model and tokenizer
# model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForQuestionAnswering.from_pretrained(model_name)

# def extract_pdf_text(pdf_path):
#     """Extract text from PDF file using PyMuPDF (fitz)."""
#     text = ""
#     doc = fitz.open(pdf_path)
#     for page_num in range(len(doc)):
#         text += doc[page_num].get_text()
#     return text

# def answer_question(question, context):
#     """Answer a question based on context using BERT model."""
#     inputs = tokenizer(question, context, return_tensors='pt', truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
#     answer_start = torch.argmax(answer_start_scores)
#     answer_end = torch.argmax(answer_end_scores) + 1
#     answer = tokenizer.convert_tokens_to_ids(inputs['input_ids'][0][answer_start:answer_end])
#     answer = tokenizer.decode(answer)
#     return answer

# def main():
#     st.title("PDF Question Answering Chatbot")

#     uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

#     if uploaded_file is not None:
#         pdf_path = uploaded_file.name
#         with open(pdf_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         pdf_text = extract_pdf_text(pdf_path)

#         st.header("PDF Content")
#         st.text(pdf_text[:500])  # Display first 500 characters of extracted text

#         st.header("Ask a Question")
#         question = st.text_input("Enter your question here")

#         if st.button("Ask"):
#             if question:
#                 answer = answer_question(question, pdf_text)
#                 st.success(f"Answer: {answer}")
#             else:
#                 st.warning("Please enter a question.")

# if __name__ == "__main__":
#     main()



# -*- coding: utf-8 -*-
"""Question and answer from PDF.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19-lBOPBoVFy6RxfliEox_T7SOJMZVbLm
"""

import os
import warnings
import PyPDF2
import nltk
import re
import json
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW, pipeline
import torch
import streamlit as st

# Ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Extracting text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Lower casing the text
def to_lower(text):
    from autocorrect import Speller
    spell = Speller(lang='en')
    texts = spell(text)
    return ' '.join([w.lower() for w in nltk.word_tokenize(texts)])

# Cleaning the text
def clean_text_basic(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stopwords_list = set(nltk.corpus.stopwords.words('english'))
    tokens = text.split()
    tokens = [token for token in tokens if token not in stopwords_list]
    text = ' '.join(tokens)
    return text

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Train the model
def train_model(training_data, model, tokenizer, epochs=8):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for example in training_data:
            context = example['context']
            question = example['question']
            answer = example['answer']
            encoded_data = tokenizer(context, question, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            start_idx = context.find(answer)
            end_idx = start_idx + len(answer)
            if start_idx == -1:
                continue
            inputs = {
                'input_ids': encoded_data['input_ids'],
                'attention_mask': encoded_data['attention_mask'],
                'start_positions': torch.tensor([start_idx]),
                'end_positions': torch.tensor([end_idx])
            }
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            if loss != loss:
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    model.save_pretrained("trained_qa_model")

# Get answer from context
def get_answer(context, question, model_path="trained_qa_model", max_seq_length=512):
    model = BertForQuestionAnswering.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoded_data = tokenizer.encode_plus(
        question,
        context,
        max_length=max_seq_length,
        truncation='only_second',
        padding='max_length',
        return_tensors='pt',
    )
    outputs = model(**encoded_data)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)
    answer_tokens = encoded_data['input_ids'][0][start_idx:end_idx+1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer

# Streamlit interface
st.title("PDF Question Answering Bot")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    pdf_path = uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text_basic(pdf_text)
    st.header("Extracted PDF Content (first 500 characters)")
    st.text(cleaned_text[:500])
    question = st.text_input("Enter your question here")
    if st.button("Ask"):
        if question:
            answer = get_answer(cleaned_text, question)
            st.success(f"Answer: {answer}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a PDF file to begin.")

# Training setup
squad_data_path = "train-v1.1.json"
with open(squad_data_path, 'r') as f:
    squad_data = json.load(f)
training_data = []
for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question = qa['question']
            if qa['answers']:
                answer = qa['answers'][0]['text']
                training_data.append({'question': question, 'answer': answer, 'context': context})
train_model(training_data[:15], model, tokenizer)


