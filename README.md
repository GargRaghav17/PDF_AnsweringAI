# PDFAnswering AI

## Introduction

Submission for ArIES Open Project 2024. This project provides a system that extracts text from PDF documents and answers user-generated questions based on the extracted content. The tool leverages Natural Language Processing (NLP) techniques to enhance document accessibility and usability, enabling users to quickly retrieve information without manually searching through the text.

## Features

- Extracts text from PDF files.
- Cleans and preprocesses the extracted text.
- Uses a pre-trained BERT model to answer questions based on the PDF content.
- Provides an interactive web interface using Streamlit.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.7 or higher
- PyMuPDF
- PyPDF2
- nltk
- transformers
- torch
- Streamlit

You can install these dependencies using pip:

```bash
pip install PyMuPDF PyPDF2 nltk transformers torch streamlit
```

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/pdf-question-answering-chatbot.git
   cd pdf-question-answering-chatbot
   ```

2. **Download NLTK resources:**

   Open a Python shell and run the following commands to download necessary NLTK resources:

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

3. **Download the pre-trained BERT model:**

   The necessary BERT model and tokenizer will be automatically downloaded the first time you run the code.

## Usage

1. **Run the Streamlit app:**

   Navigate to the project directory and run:

   ```bash
   streamlit run app.py
   ```

2. **Upload a PDF:**

   Open your browser and go to `http://localhost:8501`. You will see an interface where you can upload a PDF file.

3. **Ask a question:**

   After uploading the PDF, you can enter a question in the text input field and click the "Ask" button. The system will process your question and display the answer.

## Code Overview

### PDF Text Extraction

The system uses PyPDF2 to extract text from PDF files.

### Text Preprocessing

The text is cleaned and preprocessed using NLTK to remove noise and make it suitable for processing.

### Question Answering

A pre-trained BERT model is used to answer questions based on the extracted text.

### Streamlit Interface

The interactive web interface is built using Streamlit, allowing users to upload PDF files, input questions, and receive answers in real-time.
