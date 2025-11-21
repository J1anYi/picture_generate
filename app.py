from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import base64
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from googletrans import Translator

DATA_DIR = Path("e:/temp/picture_generate/8_floor_plan_40")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
translator = Translator()

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
            return translator.translate(text, dest="en").text
        except Exception:
            return text
    return text

docs = load_documents()
vectorstore = FAISS.from_documents(docs, embedding)

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    base64: Optional[bool] = True

@app.get("/health")
def health():
    return {"status": "ok", "documents": len(docs)}

@app.post("/search")
def search(req: QueryRequest):
    q = to_english(req.query)
    results = vectorstore.similarity_search(q, k=req.k)
    payload = []
    for d in results:
        item = {"id": d.metadata.get("id"), "description": d.page_content, "image_path": d.metadata.get("image_path")}
        if req.base64:
            try:
                with open(d.metadata.get("image_path"), "rb") as f:
                    item["image_base64"] = base64.b64encode(f.read()).decode("ascii")
            except Exception:
                item["image_base64"] = None
        payload.append(item)
    return {"query_en": q, "results": payload}
