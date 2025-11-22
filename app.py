import os
import re
import hashlib
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from config import Config
from model import ModernTrajectoryNet

# --- Configuration ---
MODEL_PATH = "checkpoints/best_model_ema.pth"
EMBEDDING_MODEL_PATH = "models/embeddinggemma-300m" # Or HuggingFace ID
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIRECT_SIM_WEIGHT = float(os.getenv("DIRECT_SIM_WEIGHT", 0.45))
KEYWORD_BOOST_WEIGHT = float(os.getenv("KEYWORD_BOOST_WEIGHT", 0.08))

app = FastAPI()

# --- Global State ---
class State:
    trajectory_model = None
    embedding_model = None
    taxonomy_cache = {} # hash(text) -> (embeddings_tensor, paths_list)

state = State()

# --- Data Models ---
class PredictionRequest(BaseModel):
    transaction: str
    taxonomy_text: str

class PredictionResponse(BaseModel):
    matches: List[Dict[str, Any]]

# --- Startup ---
@app.on_event("startup")
async def load_models():
    print("Loading Embedding Model...")
    # Fallback to a public model if local path doesn't exist
    if os.path.exists(EMBEDDING_MODEL_PATH):
        state.embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, trust_remote_code=True)
    else:
        print(f"Local model not found at {EMBEDDING_MODEL_PATH}, downloading from HF...")
        state.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") # Placeholder/Fallback
    
    print("Loading Trajectory Model...")
    cfg = Config()
    # We need to ensure the model architecture matches the checkpoint
    model = ModernTrajectoryNet(cfg)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle EMA checkpoint structure vs regular checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {MODEL_PATH}")
    else:
        print("Warning: No checkpoint found. Using random weights.")
    
    model.to(DEVICE)
    model.eval()
    state.trajectory_model = model

# --- Helper Functions ---
def parse_taxonomy(text: str) -> List[str]:
    """
    Parses indented text into a list of hierarchical path strings.
    Example Input:
    Food
        Cafe
        Restaurant
    
    Example Output:
    ["Food", "Food > Cafe", "Food > Restaurant"]
    """
    lines = text.split('\n')
    paths = []
    stack = [] # (indent_level, name)
    
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
            
        indent = len(line) - len(stripped)
        name = stripped.strip()
        
        # Pop from stack until we find the parent (lower indent)
        while stack and stack[-1][0] >= indent:
            stack.pop()
            
        stack.append((indent, name))
        
        # Construct path
        current_path = " > ".join([item[1] for item in stack])
        paths.append(current_path)
        
    return paths

def get_taxonomy_embeddings(text: str):
    """Returns (tensor [N, D], list_of_paths)"""
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    if text_hash in state.taxonomy_cache:
        return state.taxonomy_cache[text_hash]
    
    paths = parse_taxonomy(text)
    if not paths:
        return None, []
        
    print(f"Embedding {len(paths)} taxonomy nodes...")
    embeddings = state.embedding_model.encode(paths, convert_to_tensor=True, device=DEVICE)
    
    state.taxonomy_cache[text_hash] = (embeddings, paths)
    return embeddings, paths

def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def compute_keyword_boosts(transaction: str, paths: List[str]) -> List[float]:
    if KEYWORD_BOOST_WEIGHT <= 0:
        return [0.0] * len(paths)
    tx_tokens = tokenize(transaction)
    boosts = []
    for path in paths:
        cat_tokens = tokenize(path)
        if not cat_tokens or not tx_tokens:
            boosts.append(0.0)
            continue
        overlap = len(tx_tokens & cat_tokens)
        if overlap == 0:
            boosts.append(0.0)
            continue
        coverage = overlap / len(cat_tokens)
        boosts.append(KEYWORD_BOOST_WEIGHT * min(1.0, coverage * 2))
    return boosts

# --- Endpoints ---

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    if not req.transaction or not req.taxonomy_text:
        raise HTTPException(status_code=400, detail="Missing input")

    # 1. Embed Transaction
    # The model expects [B, D] or [B, L, D]. 
    # Since we are doing single-step inference, we treat the transaction as the "start" 
    # and we want to find where it "lands".
    
    # Note: The training data used raw strings.
    tx_emb = state.embedding_model.encode(req.transaction, convert_to_tensor=True, device=DEVICE)
    # Shape: [D] -> [1, D]
    tx_emb = tx_emb.unsqueeze(0)
    tx_norm = F.normalize(tx_emb, p=2, dim=1)
    
    # 2. Run Trajectory Model
    with torch.no_grad():
        # The model predicts the "delta" or the "final state" on the manifold
        predicted_final_emb = state.trajectory_model(tx_emb)
    
    # 3. Embed Taxonomy (Targets)
    target_embs, target_paths = get_taxonomy_embeddings(req.taxonomy_text)
    if target_embs is None:
         raise HTTPException(status_code=400, detail="Invalid taxonomy")

    # 4. Calculate Similarity
    # predicted: [1, D]
    # targets: [N, D]
    
    pred_norm = F.normalize(predicted_final_emb, p=2, dim=1)
    targets_norm = F.normalize(target_embs, p=2, dim=1)
    pred_sims = torch.mm(pred_norm, targets_norm.T).squeeze(0)
    direct_sims = torch.mm(tx_norm, targets_norm.T).squeeze(0)

    if DIRECT_SIM_WEIGHT > 0:
        sims = (1 - DIRECT_SIM_WEIGHT) * pred_sims + DIRECT_SIM_WEIGHT * direct_sims
    else:
        sims = pred_sims
    
    # Cosine similarity: [1, N]
    keyword_boosts = compute_keyword_boosts(req.transaction, target_paths)
    if any(keyword_boosts):
        boost_tensor = torch.tensor(keyword_boosts, device=sims.device, dtype=sims.dtype)
        sims = sims + boost_tensor
    
    # 5. Rank
    top_k = min(10, len(target_paths))
    scores, indices = torch.topk(sims, top_k)
    
    matches = []
    for score, idx in zip(scores, indices):
        matches.append({
            "category": target_paths[idx.item()],
            "score": float(score.item())
        })
        
    return {"matches": matches}

# Serve Static Files (Frontend)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
