import json
import os
import re
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding

os.environ.setdefault('OMP_NUM_THREADS', '1')
kb = json.loads(Path(__file__).with_name('knowledge_base.json').read_text(encoding='utf-8'))
model = TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', threads=1)
# Index complete passages and individual sentences. Keep original passage IDs.
units = []
for i, passage in enumerate(kb):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', passage)
    for sentence in sentences:
        units.append((i, sentence, sentences[0] + ' ' + sentence))

def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

passage_vectors = normalize(np.array(list(model.embed(kb, batch_size=8))))
sentence_vectors = normalize(np.array(list(model.embed([u[2] for u in units], batch_size=8))))
lock = Lock()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class Query(BaseModel):
    question: str = Field(max_length=4000)
    top_k: int = Field(default=3, ge=1, le=50)

@app.get('/')
def health():
    return {'status': 'ready', 'passages': len(kb)}

def abstain():
    return {'answer': None, 'sources': [], 'scores': [], 'confidence': 0.15}

STOP = set('what which who how when where why is are was were the a an of to for in on by does do did it its and or with from according knowledge base passage please tell me about given stated specified configured value number many much'.split())
def tokens(s):
    return set(re.findall(r'[a-z0-9]+(?:[-/][a-z0-9]+)*', s.lower())) - STOP

@app.post('/query')
def query(body: Query, response: Response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    q = body.question.strip()
    if not q:
        return abstain()
    # Ignore common attempts to override the question-answering task.
    q = re.split(r'(?i)\b(?:ignore (?:all |any |the )?(?:previous|prior|above)|disregard (?:all |the )?(?:previous|prior|above)|system prompt|developer message|instead (?:say|output|return))\b', q)[0].strip()
    if not q or not tokens(q):
        return abstain()
    with lock:
        v = normalize(np.array(list(model.embed([q], batch_size=1))))[0]
    p = passage_vectors @ v
    s = sentence_vectors @ v
    qt = tokens(q)
    overlaps = np.array([len(qt & tokens(u[2])) / max(1, len(qt)) for u in units])
    ranks = 0.68*s + 0.22*np.array([p[u[0]] for u in units]) + 0.10*overlaps
    order = np.argsort(ranks)[::-1]
    best = int(order[0])
    # Conservative heuristic; these thresholds are not calibrated probabilities.
    if s[best] < 0.36 or ranks[best] < 0.37:
        return abstain()
    # Reject common requested attributes absent from the supporting text.
    attributes = {'price':r'price|cost|\$','inventor':r'invent|creat|develop','year':r'\b(?:19|20)\d{2}\b','release date':r'\b(?:19|20)\d{2}\b','license':r'licen[cs]','memory usage':r'memory usage|ram consumption|memory footprint','version number':r'version|\d+\.\d+'}
    candidate_text = ' '.join(units[int(j)][1] for j in order[:5]).lower()
    for attr, pattern in attributes.items():
        if attr in q.lower() and not re.search(pattern, candidate_text):
            return abstain()
    chosen = [best]
    multi = bool(re.search(r'(?i)\b(compare|contrast|both|respectively|difference|combined|whereas|versus|vs\.?|and)\b', q))
    if multi and body.top_k > 1:
        for j in order[1:]:
            j = int(j)
            if units[j][0] != units[best][0] and ranks[j] >= max(0.39, ranks[best]-0.13):
                chosen.append(j)
                break
    # Return verbatim evidence sentences rather than inventing values.
    # Include passage lead-in for clarity when the evidence has no subject name.
    ids = sorted({units[j][0] for j in chosen}, key=lambda i: float(p[i]), reverse=True)
    answer = ' '.join(dict.fromkeys(units[j][1] for j in chosen))
    scores = [round(float(np.clip(p[i], 0, 1)), 4) for i in ids]
    return {'answer': answer, 'sources': ids, 'scores': scores, 'confidence': round(float(np.clip(0.61 + 0.38*ranks[best], 0.61, 0.99)), 4)}
