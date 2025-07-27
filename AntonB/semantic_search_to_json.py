"""
Library for building a Chroma vectorstore from PDFs and performing semantic searches,
then formatting results as structured JSON.
"""
import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any, Union, Optional

from fastembed import TextEmbedding
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FastEmbedEmbeddings:
    """
    Embedding wrapper using fastembed.TextEmbedding
    """
    def __init__(self):
        self.model = TextEmbedding()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return list(self.model.embed(texts))

    def embed_query(self, text: str) -> List[float]:
        return list(self.model.embed([text]))[0]


def get_embedding_function() -> FastEmbedEmbeddings:
    """Return a fresh embeddings instance"""
    return FastEmbedEmbeddings()


def load_and_split_documents(
    data_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 80
) -> List[Document]:
    """
    Load all PDFs under data_path, split into chunks, and return list of Documents.
    """
    documents: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    for fname in sorted(os.listdir(data_path)):
        if not fname.lower().endswith('.pdf'):
            continue
        full = os.path.join(data_path, fname)
        md_pages = __import__('pymupdf4llm').to_markdown(full, page_chunks=True)
        for i, page in enumerate(md_pages, start=1):
            documents.append(
                Document(
                    page_content=page['text'],
                    metadata={'source': fname, 'page': i}
                )
            )
    return splitter.split_documents(documents)


def calculate_chunk_ids(chunks: List[Document]) -> None:
    """
    Mutates each Document in-place: adds a unique 'id' metadata field.
    """
    last_id: Optional[str] = None
    idx = 0
    for chunk in chunks:
        src = chunk.metadata['source']
        pg = chunk.metadata['page']
        current = f"{src}:{pg}"
        idx = idx + 1 if current == last_id else 0
        chunk.metadata['id'] = f"{current}:{idx}"
        last_id = current


def build_chroma(
    chunks: List[Document],
    persist_directory: str = 'chroma',
    reset: bool = False
) -> Chroma:
    """
    Build or update a Chroma vectorstore from the given chunks.
    If reset=True, deletes persist_directory first.
    Returns the Chroma instance.
    """
    if reset and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=get_embedding_function())
    calculate_chunk_ids(chunks)
    existing = set(db.get().get('ids', []))
    new_chunks = [c for c in chunks if c.metadata['id'] not in existing]
    if new_chunks:
        ids = [c.metadata['id'] for c in new_chunks]
        db.add_documents(documents=new_chunks, ids=ids)
    return db


def query_and_format(
    db: Chroma,
    input_meta: Dict[str, Any],
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Execute semantic search for `query`, returning structured JSON.
    `input_meta` should contain keys:
      - documents: list of filenames or dicts with 'filename'
      - persona: dict with 'role'
      - job_to_be_done: dict with 'task'
    """
    results = db.similarity_search_with_score(query, k=top_k)
    # normalize input_documents
    raw = input_meta.get('documents', [])
    docs_list = []
    for item in raw:
        if isinstance(item, dict) and 'filename' in item:
            docs_list.append(item['filename'])
        elif isinstance(item, str):
            docs_list.append(item)
    # build output
    output = {
        'metadata': {
            'input_documents': docs_list,
            'persona': input_meta.get('persona', {}).get('role', ''),
            'job_to_be_done': input_meta.get('job_to_be_done', {}).get('task', ''),
            'processing_timestamp': datetime.utcnow().isoformat()
        },
        'extracted_sections': [],
        'subsection_analysis': []
    }
    # extracted_sections
    for rank, (doc, _) in enumerate(results, start=1):
        lines = doc.page_content.splitlines()
        title = lines[0].strip() if lines else ''
        output['extracted_sections'].append({
            'document': doc.metadata['source'],
            'section_title': title,
            'importance_rank': rank,
            'page_number': doc.metadata.get('page')
        })
    # subsection_analysis
    for doc, _ in results:
        text = doc.page_content.strip()
        output['subsection_analysis'].append({
            'document': doc.metadata['source'],
            'refined_text': text[:200],
            'page_number': doc.metadata.get('page')
        })
    return output

def Output_Analysis(query, input_json):
    chunks = load_and_split_documents('data')
    db = build_chroma(chunks, 'chroma', reset=True)
    spec = json.load(open(input_json))
    result = query_and_format(db, spec, query)
    print(json.dumps(result, indent=4))
    with open('output.json', 'w') as f:
        json.dump(result, f, indent=4)
