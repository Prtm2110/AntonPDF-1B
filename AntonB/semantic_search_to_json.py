import argparse
import json
import os
import shutil
from datetime import datetime

from fastembed import TextEmbedding
from typing import List, Dict, Any, Union
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma

# Embedding helper
class FastEmbedEmbeddings:
    def __init__(self):
        self.model = TextEmbedding()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return list(self.model.embed(texts))
    
    def embed_query(self, text: str) -> List[float]:
        return list(self.model.embed([text]))[0]

def get_embedding_function():
    return FastEmbedEmbeddings()

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Load and split PDFs into Documents
def load_and_split_documents() -> List[Document]:
    documents: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    for file in sorted(os.listdir(DATA_PATH)):
        if file.endswith('.pdf'):
            path = os.path.join(DATA_PATH, file)
            md_pages = __import__('pymupdf4llm').to_markdown(path, page_chunks=True)
            for i, page in enumerate(md_pages):
                text = page['text']
                metadata = {'source': file, 'page': i + 1}
                documents.append(Document(page_content=text, metadata=metadata))
    return text_splitter.split_documents(documents)

# Assign unique IDs
def calculate_chunk_ids(chunks: List[Document]) -> List[Document]:
    last_page_id = None
    idx = 0
    for chunk in chunks:
        src = chunk.metadata['source']
        pg = chunk.metadata['page']
        current = f"{src}:{pg}"
        if current == last_page_id:
            idx += 1
        else:
            idx = 0
        chunk.metadata['id'] = f"{current}:{idx}"
        last_page_id = current
    return chunks

# Build/update Chroma DB
def build_chroma(chunks: List[Document], reset: bool = False) -> Chroma:
    if reset and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks = calculate_chunk_ids(chunks)
    existing_items = db.get()
    existing_ids = set(existing_items.get('ids', []))
    to_add = [c for c in chunks if c.metadata['id'] not in existing_ids]
    if to_add:
        ids = [c.metadata['id'] for c in to_add]
        db.add_documents(to_add, ids=ids)
    return db

# Perform semantic search and format results
def query_and_format(
    db: Chroma,
    input_meta: Dict[str, Any],
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    results = db.similarity_search_with_score(query, k=top_k)
    # Prepare input_documents list defensively
    raw_docs: List[Union[str, Dict[str, Any]]] = input_meta.get('documents', [])  # might be list of strings or dicts
    input_docs: List[str] = []
    for item in raw_docs:
        if isinstance(item, str):
            input_docs.append(item)
        elif isinstance(item, dict) and 'filename' in item:
            input_docs.append(item['filename'])
        else:
            input_docs.append(str(item))
    # Build output
    out: Dict[str, Any] = {
        'metadata': {
            'input_documents': input_docs,
            'persona': input_meta.get('persona', {}).get('role', ''),
            'job_to_be_done': input_meta.get('job_to_be_done', {}).get('task', ''),
            'processing_timestamp': datetime.utcnow().isoformat()
        }
    }
    # Extracted sections
    extracted: List[Dict[str, Any]] = []
    for rank, (doc, _) in enumerate(results, start=1):
        lines = doc.page_content.strip().splitlines()
        title = lines[0] if lines else ''
        extracted.append({
            'document': doc.metadata['source'],
            'section_title': title,
            'importance_rank': rank,
            'page_number': doc.metadata.get('page')
        })
    out['extracted_sections'] = extracted
    # Subsection analysis
    subs: List[Dict[str, Any]] = []
    for doc, _ in results:
        subs.append({
            'document': doc.metadata['source'],
            'refined_text': doc.page_content[:200].strip(),
            'page_number': doc.metadata.get('page')
        })
    out['subsection_analysis'] = subs
    return out

# Entrypoint
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='Reset DB')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON')
    parser.add_argument('--query', type=str, required=True, help='Search query')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        input_meta = json.load(f)
    docs = load_and_split_documents()
    db = build_chroma(docs, reset=args.reset)
    result = query_and_format(db, input_meta, args.query)
    print(json.dumps(result, indent=4))
