import streamlit as st
import os
import numpy as np
from gensim.models import KeyedVectors

# Load model
model = KeyedVectors.load_word2vec_format("word2vec.txt")

# Load resumes
def load_resumes(folder="resumes"):
    resumes = {}
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                resumes[file] = f.read()
    return resumes

# Convert sentence to vector
def sentence_vector(text):
    words = text.lower().split()
    vectors = [model[w] for w in words if w in model]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Cosine similarity
def similarity(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# UI
st.title("📄 Resume Matching System (Word2Vec)")

job_desc = st.text_area("Enter Job Description")

top_n = st.slider("Top Results", 1, 10, 5)

if st.button("Find Best Resumes"):
    resumes = load_resumes()
    job_vec = sentence_vector(job_desc)

    scores = []

    for name, text in resumes.items():
        res_vec = sentence_vector(text)
        score = similarity(job_vec, res_vec)
        scores.append((name, score, text))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    st.subheader("🏆 Top Matching Resumes")

    for name, score, text in scores[:top_n]:
        st.write(f"**{name}** → Score: {score:.3f}")
        st.write(text)
        st.write("---")