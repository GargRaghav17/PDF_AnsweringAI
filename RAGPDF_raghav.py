# import fitz  # PyMuPDF
# from transformers import BertTokenizer, BertForQuestionAnswering
# import torch

# # Initialize the tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# def extract_text_from_pdf(pdf_path):
#     document = fitz.open(pdf_path)
#     text = ""
#     for page_num in range(len(document)):
#         try:
#             page = document.load_page(page_num)
#             page_text = page.get_text()
#             if page_text:  # Check if the page has text
#                 text += page_text
#             else:
#                 print(f"Page {page_num + 1} has no extractable text.")
#         except Exception as e:
#             print(f"Error reading page {page_num + 1}: {e}")
#     return text

# def answer_question(question, text):
#     inputs = tokenizer(question, text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
#     with torch.no_grad():
#         outputs = model(**inputs)
#     answer_start = torch.argmax(outputs.start_logits)
#     answer_end = torch.argmax(outputs.end_logits) + 1
#     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
#     return answer

# if __name__ == '__main__':
#     # Specify the path to your PDF file
#     pdf_path = "c:/Users/hp/Desktop/engbookmerged.pdf"
    
#     # Extract text from the PDF
#     pdf_text = extract_text_from_pdf(pdf_path)
#     print("Extracted Text:")
#     print(pdf_text[:5000])  # Print the first 1000 characters to avoid too much output
    
#     # Ask a question about the PDF
#     question = "how did the birds fall?"
#     answer = answer_question(question, pdf_text)
    
#     print("\nQuestion:")
#     print(question)
#     print("\nAnswer:")
#     print(answer)

import streamlit as st
import os
import fitz  # PyMuPDF for PDF text extraction
from transformers import pipeline  # Hugging Face transformers for question answering
import nltk

# Download NLTK resources (needed for text processing)
nltk.download('punkt')

# Initialize the question-answering pipeline from transformers
qa_pipeline = pipeline("question-answering")

def extract_pdf_text(pdf_path):
    """Extract text from PDF file using PyMuPDF (fitz)."""
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        text += doc[page_num].get_text()
    return text

def answer_question(question, context):
    """Answer a question based on context using transformers pipeline."""
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def main():
    st.title("PDF Question Answering Chatbot")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        pdf_path = os.path.join("uploads", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_text = extract_pdf_text(pdf_path)

        st.header("PDF Content")
        st.text(pdf_text[:500])  # Display first 500 characters of extracted text

        st.header("Ask a Question")
        question = st.text_input("Enter your question here")

        if st.button("Ask"):
            if question:
                answer = answer_question(question, pdf_text)
                st.success(f"Answer: {answer}")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
