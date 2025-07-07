import numpy as np
from transformers import pipeline

qa_model = pipeline("text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")

def get_most_relevant_chunks(query, index, embeddings, chunks, embed_model):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), k=3)
    return [chunks[i] for i in I[0]]

def answer_question(query, index, embeddings, chunks, embed_model):
    top_chunks = get_most_relevant_chunks(query, index, embeddings, chunks, embed_model)
    context = "\n".join(top_chunks)
    prompt = f"Answer this question based on the context below:\n\nContext: {context}\n\nQuestion: {query}"
    response = qa_model(prompt, max_new_tokens=100)[0]['generated_text']
    return response
