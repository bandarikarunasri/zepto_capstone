import os, glob
from typing import List, TypedDict
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
_encoder = SentenceTransformer(MODEL_NAME)

# Simple in-memory store instead of Chroma - works on Python 3.14
_store = [] # list of {id, content, vector}

def load_corpus_if_needed():
    if len(_store) >= 8:
        return
    for fp in sorted(glob.glob("./docs/doc_*.txt")):
        raw = open(fp, "r", encoding="utf-8").read()
        did = os.path.basename(fp).split(".")[0]
        vec = _encoder.encode(raw)
        _store.append({"id": did, "content": raw, "vector": vec})

class QueryIn(BaseModel):
    query: str
class QueryOut(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class State(TypedDict):
    query: str
    intent: str
    docs: List[dict]
    answer: str
    sources: List[str]
    confidence: float

_POLICY_KEYS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift", "support hours", "fee", "pass"]

def node_classify(state: State):
    q_low = state["query"].lower()
    is_policy = any(k in q_low for k in _POLICY_KEYS)
    label = "policy_question" if is_policy else "general_question"
    return {"intent": label}

def cosine(a,b):
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def node_retrieve_answer(state: State):
    q_vec = _encoder.encode(state["query"])
    scored = []
    for item in _store:
        s = cosine(q_vec, item["vector"])
        scored.append((s, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    top3 = [x[1] for x in scored[:3]]
    top_ids = [x["id"] for x in top3]
    top_text = top3[0]["content"][:250] if top3 else ""
    canned = f"Based on the retrieved context: {top_text}"
    return {"answer": canned, "sources": top_ids, "confidence": 1.0, "docs": top3}

def node_direct(state: State):
    return {"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 1.0}

def router(state: State):
    return state["intent"]

g = StateGraph(State)
g.add_node("classify_intent", node_classify)
g.add_node("retrieve_and_answer", node_retrieve_answer)
g.add_node("direct_answer", node_direct)
g.set_entry_point("classify_intent")
g.add_conditional_edges("classify_intent", router, {"policy_question": "retrieve_and_answer", "general_question": "direct_answer"})
g.add_edge("retrieve_and_answer", END)
g.add_edge("direct_answer", END)
my_graph = g.compile()

app = FastAPI(title="Zepto Assistant")

@app.on_event("startup")
def on_start():
    load_corpus_if_needed()

@app.post("/ask", response_model=QueryOut)
def ask_api(inp: QueryIn):
    out = my_graph.invoke({"query": inp.query})
    return QueryOut(answer=out["answer"], sources=out.get("sources", []), confidence=out.get("confidence", 1.0))

@app.get("/")
def health():
    return {"ok": True, "mode": "mock_no_chroma"}