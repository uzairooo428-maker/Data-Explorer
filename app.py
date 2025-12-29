import os
import gdown
import faiss
import numpy as np
import pandas as pd
import gradio as gr

from sentence_transformers import SentenceTransformer
from groq import Groq

# =====================================================
# 🔐 GROQ API KEY (Hugging Face Spaces)
# =====================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Add it in Hugging Face Space → Settings → Secrets."
    )

client = Groq(api_key=GROQ_API_KEY)

# =====================================================
# 📥 DOWNLOAD CSV KNOWLEDGE BASE FROM GOOGLE DRIVE
# =====================================================
CSV_URL = "https://drive.google.com/uc?id=1KLFGbFmDlRBfVcEJQbCYUspcm7HAk5ym"
CSV_PATH = "knowledge_base.csv"

if not os.path.exists(CSV_PATH):
    gdown.download(CSV_URL, CSV_PATH, quiet=False)

# =====================================================
# 📊 LOAD & PREPARE DATA
# =====================================================
df = pd.read_csv(CSV_PATH)

documents = df.astype(str).apply(
    lambda row: " | ".join(row.values),
    axis=1
).tolist()

# =====================================================
# 🔢 EMBEDDINGS + FAISS INDEX
# =====================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# =====================================================
# ❓ RAG QUERY FUNCTION (NO HALLUCINATION)
# =====================================================
SIMILARITY_THRESHOLD = 1.0  # lower = stricter

def ask_question(question: str) -> str:
    query_embedding = embedder.encode([question])
    distances, indices = index.search(query_embedding, k=3)

    # If question is outside KB
    if distances[0][0] > SIMILARITY_THRESHOLD:
        return (
            "### ❌ Answer\n"
            "**I do not know.**  \n"
            "Your question is not related to the available knowledge base."
        )

    context = "\n".join([documents[i] for i in indices[0]])

    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say exactly: "I do not know."
Context:
{context}
Question:
{question}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = completion.choices[0].message.content

    return f"### ✅ Answer\n{answer}"

# =====================================================
# 🎨 GEN-Z VIBRANT UI
# =====================================================
css = """
body {
    background: linear-gradient(135deg, #fdfbfb, #ebedee);
}
h1 {
    text-align: center;
    color: #6a0dad;
    font-size: 36px;
}
.answer-box {
    border-radius: 16px;
    border: 3px solid #6a0dad;
    padding: 20px;
    background: white;
    font-size: 16px;
}
.gr-button {
    background: linear-gradient(45deg, #6a0dad, #ff6f61);
    color: white;
    font-weight: bold;
    border-radius: 14px;
}
"""

# =====================================================
# 🚀 GRADIO APP
# =====================================================
with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1>✨ CSV Knowledge-Base RAG Assistant</h1>")
    gr.Markdown(
        "Ask questions **only related to the CSV knowledge base**. "
        "Unrelated questions will return *I do not know*."
    )

    question = gr.Textbox(
        label="Your Question",
        placeholder="Ask something from the CSV..."
    )

    answer = gr.Markdown(elem_classes="answer-box")

    btn = gr.Button("🚀 Get Answer")

    btn.click(
        fn=ask_question,
        inputs=question,
        outputs=answer
    )

demo.launch()
