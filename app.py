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

    # 🧠 Q&A history
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    question = st.text_input("Ask a question about the PDF:")
    if question:
        with st.spinner("Generating answer..."):
            answer = answer_question(question, index, embeddings, chunk_data, model)
            st.session_state.qa_history.append((question, answer))
            st.success(answer)

    # Show & Download QnA
    if st.session_state.qa_history:
        st.subheader("📘 Q&A History")
        for q, a in st.session_state.qa_history:
            st.markdown(f"**Q:** {q}\n\n**A:** {a}\n---")

        all_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.qa_history])
        st.download_button("📥 Download Answers", all_text, file_name="qa_answers.txt")
