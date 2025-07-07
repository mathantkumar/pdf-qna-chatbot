import streamlit as st
from pdf_reader import extract_text_from_pdf
from embedder import chunk_text, create_vector_store, model
from qna import answer_question

st.title("📄 PDF QnA Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Extracting text and creating embeddings...")
    text = extract_text_from_pdf("temp.pdf")
    chunks = chunk_text(text)
    index, embeddings, chunk_data = create_vector_store(chunks)

    question = st.text_input("Ask a question about the PDF:")
    if question:
        with st.spinner("Generating answer..."):
            answer = answer_question(question, index, embeddings, chunk_data, model)
            st.success(answer)
