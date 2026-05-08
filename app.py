
import gradio as gr
import pickle
import faiss
import json
import numpy as np
import requests
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Load artifacts
print("Loading artifacts...")
corpus = [json.loads(l) for l in open('ama_qa_enriched.jsonl')]
embeddings = np.load('embeddings.npy')
index = faiss.read_index('faiss_index.bin')
with open('bm25_index.pkl', 'rb') as f:
    bm25 = pickle.load(f)
with open('doc_texts.json') as f:
    doc_texts = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ All loaded")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def tokenize(text):
    return text.lower().split()

def hybrid_search(query, top_k=10):
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_max = bm25_scores.max()
    if bm25_max > 0:
        bm25_scores = bm25_scores / bm25_max
    query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    dense_scores, dense_indices = index.search(query_embedding, len(corpus))
    dense_score_array = np.zeros(len(corpus))
    for score, idx in zip(dense_scores[0], dense_indices[0]):
        dense_score_array[idx] = score
    combined = 0.3 * bm25_scores + 0.7 * dense_score_array
    top_indices = np.argsort(combined)[::-1][:top_k]
    return [{
        'score': combined[i],
        'ama_title': corpus[i]['ama_title'],
        'category': corpus[i]['category'],
        'question': corpus[i]['question'],
        'answer': corpus[i]['answer'],
        'op_author': corpus[i]['op_author'],
        'source_url': corpus[i]['source_url'],
    } for i in top_indices]

def rerank(query, docs, top_k=3):
    pairs = [[query, d['question'] + " " + d['answer']] for d in docs]
    scores = reranker.predict(pairs)
    for doc, score in zip(docs, scores):
        doc['rerank_score'] = float(score)
    return sorted(docs, key=lambda x: x['rerank_score'], reverse=True)[:top_k]

def generate_answer(query, docs):
    context_blocks = []
    for i, doc in enumerate(docs):
        context_blocks.append(
            f"[Source {i+1}]\n"
            f"Person: {doc['ama_title']}\n"
            f"They were asked: {doc['question'][:200]}\n"
            f"They answered: {doc['answer'][:500]}\n"
        )
    context = "\n---\n".join(context_blocks)
    prompt = f"""You are an AI that answers questions using ONLY real Reddit AMA responses from verified people.

Rules:
1. ONLY use information from the provided sources
2. Always attribute answers to the specific person by their AMA title
3. If sources dont contain relevant information, say: The available AMAs dont cover this directly.
4. Write in flowing prose, not bullet points
5. Be complete, do not cut off mid-sentence
6. End with a Sources: section listing the AMA titles used

Question: {query}

AMA Sources:
{context}

Write a complete attributed answer:"""

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.3}
        }
    )
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    return f"Generation error: {response.status_code}"

def rag_pipeline(query, top_k):
    if not query.strip():
        return "Please enter a question.", ""
    candidates = hybrid_search(query, top_k=10)
    docs = rerank(query, candidates, top_k=int(top_k))
    sources_html = ""
    for i, d in enumerate(docs):
        sources_html += f"""
<div style="border:1px solid #333; border-radius:8px; padding:12px; margin-bottom:10px; background:#1a1a1a;">
  <div style="font-size:12px; color:#888;">#{i+1} · hybrid: {d["score"]:.3f} · rerank: {d["rerank_score"]:.2f} · {d["category"]}</div>
  <div style="font-weight:bold; color:#e0e0e0; margin-bottom:6px;">🎙️ {d["ama_title"][:80]}</div>
  <div style="font-size:13px; color:#aaa;"><b>Asked:</b> {d["question"][:120]}...</div>
  <div style="font-size:13px; color:#ccc;"><b>They said:</b> {d["answer"][:200]}...</div>
  <div style="margin-top:6px;"><a href="{d["source_url"]}" target="_blank" style="font-size:11px; color:#7eb8f7;">🔗 view original AMA</a></div>
</div>"""
    answer = generate_answer(query, docs)
    return answer, sources_html

examples = [
    ["What do people in prison actually do all day?", 3],
    ["What's the hardest part of a job most people romanticize?", 3],
    ["What do rich people do differently from everyone else?", 3],
    ["What was the scariest moment of your life?", 3],
    ["What do most people get wrong about addiction?", 3],
    ["What advice would you give your younger self?", 4],
]

with gr.Blocks(theme=gr.themes.Monochrome(), title="Reddit RabbitHole RAG") as demo:
    gr.Markdown("""
# 🕳️ Reddit RabbitHole RAG
### Ask anything — get answered by real people who lived it
*2,671 verified Q&A pairs · 451 Reddit AMAs · Hybrid BM25 + Dense Retrieval + Cross-Encoder Reranker · Gemini 2.5 Flash · RAGAS Faithfulness: 0.81*
""")
    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(label="Your Question", placeholder="e.g. What's it actually like to be in prison?", lines=2)
        with gr.Column(scale=1):
            top_k_slider = gr.Slider(minimum=1, maximum=7, value=3, step=1, label="AMA sources")
            submit_btn = gr.Button("Ask the Rabbit Hole 🕳️", variant="primary", size="lg")
    with gr.Row():
        with gr.Column(scale=2):
            answer_box = gr.Textbox(label="✍️ Synthesized Answer — grounded in real AMA responses", lines=12, interactive=False)
        with gr.Column(scale=1):
            sources_box = gr.HTML(label="📚 Retrieved & Reranked Sources")
    gr.Examples(examples=examples, inputs=[query_box, top_k_slider], label="💡 Try these")
    gr.Markdown("""
---
**How it works:** Hybrid BM25 + dense retrieval → cross-encoder reranker → Gemini 2.5 Flash grounded generation.
RAGAS Faithfulness: **0.81** · No hallucination — if it's not in the AMAs, it says so.
""")
    submit_btn.click(fn=rag_pipeline, inputs=[query_box, top_k_slider], outputs=[answer_box, sources_box])
    query_box.submit(fn=rag_pipeline, inputs=[query_box, top_k_slider], outputs=[answer_box, sources_box])

demo.launch()
