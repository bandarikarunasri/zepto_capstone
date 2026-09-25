# Zepto Support Assistant - RAG Pipeline

Ingestion -> Embedding -> Retrieval -> Generation

1. ingestion: main.py load_corpus_if_needed() docs/*.txt chaduvuthundi. Startup lo run avuthundi.

2. embedding: all-MiniLM-L6-v2 model 384 dim vector. ChromaDB collection zepto_docs_v2 lo save.

3. retrieval: retrieve_and_answer node lo query embed chesi top-3 cosine similarity tho teestha. Rendu modes lo real ga run avuthundi.

4. generation: classify_intent -> conditional edge -> retrieve_and_answer / direct_answer

MOCK_LLM toggle:
mock=1 (graded): keyword heuristic (delivery, return etc) + canned "Based on the retrieved context:..."
real=0 (optional): Groq LLM + MY_PROMPT template + retry 3 times

Run:
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

Example:
POST /ask {"query": "delivery fee below 149?"} -> returns doc_01 context
POST /ask {"query": "tell joke"} -> fixed string

Docker:
docker build -t zepto .
docker run -p 7860:7860 zepto