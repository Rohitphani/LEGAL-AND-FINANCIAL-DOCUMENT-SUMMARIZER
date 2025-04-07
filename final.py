# Install dependencies if not already installed
# Run the following commands:
# pip install pypdf2 fpdf torch transformers streamlit

import streamlit as st
import torch
from PyPDF2 import PdfReader
from fpdf import FPDF
from transformers import BartTokenizer, BartForConditionalGeneration, PegasusForConditionalGeneration, PegasusTokenizer

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the models and tokenizers
@st.cache_resource
def load_models():
    bart_model = BartForConditionalGeneration.from_pretrained("./bart1/fine_tuned_bart_model").to(device)
    bart_tokenizer = BartTokenizer.from_pretrained("./bart1/fine_tuned_bart_model")

    pegasus_model = PegasusForConditionalGeneration.from_pretrained("./fine_tuned_pegasus_model").to(device)
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("./fine_tuned_pegasus_model")

    return {
        "Financial": (bart_model, bart_tokenizer),
        "Legal": (pegasus_model, pegasus_tokenizer),
    }

models = load_models()

# Function to summarize text
def summarize_text(input_text, model, tokenizer, min_length=100, max_length=300):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Function to generate a PDF file
def generate_pdf(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    return pdf

# Streamlit UI
st.title("Document Summarizer")

# Dropdown to select summarizer type
summarizer_type = st.selectbox("Select Summarizer Type", ["Financial", "Legal"])

# PDF file uploader
uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

if uploaded_pdf is not None:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_pdf)
    
    if extracted_text.strip():
        #st.subheader("Extracted Text:")
        #st.write(extracted_text)

        if st.button("Summarize"):
            with st.spinner("Generating summary..."):
                model, tokenizer = models[summarizer_type]
                min_length = 100
                max_length = 500 if summarizer_type == "Financial" else 300
                summary = summarize_text(extracted_text, model, tokenizer, min_length, max_length)
            
            st.subheader("Summary:")
            st.write(summary)

            # Generate the PDF
            pdf = generate_pdf(summary)
            pdf_output = pdf.output(dest="S").encode("latin1")  # Save PDF to a byte string

            # Provide a download button for the PDF
            st.download_button(
                label="Download Summary as PDF",
                data=pdf_output,
                file_name="summary.pdf",
                mime="application/pdf"
            )
    else:
        st.error("The uploaded PDF does not contain extractable text.")
