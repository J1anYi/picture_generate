from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import os
from datetime import datetime
import shutil
import time
import random
import base64
import re
from fastapi.responses import FileResponse
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from deep_translator import GoogleTranslator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "8_floor_plan_40")))

EMB = None
VS = None
DOCS: List[Document] = []
translator = GoogleTranslator(source="auto", target="en")

def load_documents() -> List[Document]:
    docs: List[Document] = []
    for txt_path in DATA_DIR.glob("*.txt"):
        stem = txt_path.stem
        if not stem.isdigit():
            continue
        png_path = DATA_DIR / f"{stem}.png"
        if not png_path.exists():
            alt_png = DATA_DIR / "images" / f"{stem}.png"
            if alt_png.exists():
                png_path = alt_png
            else:
                continue
        content = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        docs.append(Document(page_content=content, metadata={"image_path": str(png_path), "id": stem}))
    return docs

def is_chinese(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None

def to_english(text: str) -> str:
    if not text:
        return text
    if is_chinese(text):
        try:
            return translator.translate(text)
        except Exception:
            return text
    return text

def get_vectorstore():
    global VS, EMB, DOCS
    if VS is not None:
        return VS
    if not DOCS:
        DOCS = load_documents()
    model_path = os.getenv("EMBEDDING_LOCAL_DIR")
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L6-v2")
    try:
        EMB = HuggingFaceEmbeddings(model_name=model_path or model_name)
        VS = FAISS.from_documents(DOCS, EMB)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"embedding init failed: {e}")
    return VS

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    base64: Optional[bool] = True

@app.get("/health")
def health():
    return {"status": "ok", "documents": len(DOCS) if DOCS else len(load_documents())}

@app.post("/search")
def search(req: QueryRequest):
    q = to_english(req.query)
    results = get_vectorstore().similarity_search(q, k=req.k)
    date_dir = BASE_DIR / "temp" / datetime.now().strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for d in results:
        item = {"id": d.metadata.get("id"), "description": d.page_content, "image_path": d.metadata.get("image_path")}
        if req.base64:
            try:
                with open(d.metadata.get("image_path"), "rb") as f:
                    item["image_base64"] = base64.b64encode(f.read()).decode("ascii")
            except Exception:
                item["image_base64"] = None
        try:
            src = Path(d.metadata.get("image_path"))
            dst = date_dir / src.name
            if src.exists():
                if not dst.exists():
                    shutil.copyfile(src, dst)
                item["image_path"] = str(dst)
        except Exception:
            item["image_path"] = str(Path(d.metadata.get("image_path")).resolve())
        payload.append(item)
    time.sleep(random.randint(5, 10))
    return {"query_en": q, "results": payload}

def copy_to_date_dir(src: Path) -> Path:
    date_dir = BASE_DIR / "temp" / datetime.now().strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    dst = date_dir / src.name
    if src.exists() and not dst.exists():
        shutil.copyfile(src, dst)
    return dst

def resolve_image_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return path

@app.get("/image")
def get_image(path: str):
    file_path = resolve_image_path(path)
    return FileResponse(str(file_path), media_type="image/png")

@app.get("/search_image")
def search_image(query: str, k: int = 1, index: int = 0):
    q = to_english(query)
    results = get_vectorstore().similarity_search(q, k=k)
    if not results:
        raise HTTPException(status_code=404, detail="no result")
    index = max(0, min(index, len(results) - 1))
    src = Path(results[index].metadata.get("image_path"))
    dst = copy_to_date_dir(src)
    time.sleep(random.randint(5, 10))
    return FileResponse(str(dst), media_type="image/png")