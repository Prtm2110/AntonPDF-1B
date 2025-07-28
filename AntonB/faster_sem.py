import json
import os
import shutil
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import pymupdf  # PyMuPDF for plain-text extraction
from fastembed import TextEmbedding
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_embedding_function():
    """
    Returns an embedding function object compatible with Chroma,
    wrapping fastembed.TextEmbedding to provide `embed_documents` and `embed_query` methods.
    """
    class EmbeddingWrapper:
        def __init__(self):
            self.model = TextEmbedding()

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            # fastembed.TextEmbedding.embed returns a generator -> convert to list
            return list(self.model.embed(texts))

        def embed_query(self, text: str) -> List[float]:
            # embed single text, returns generator inside a list
            return list(self.model.embed([text]))[0]

    return EmbeddingWrapper()


def load_and_split_documents(
    data_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 80
) -> List[Document]:
    """
    Load all PDF files from the given directory, extract plain text per page,
    then split into chunks using the provided splitter settings.
    """
    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    for fname in sorted(os.listdir(data_path)):
        if not fname.lower().endswith('.pdf'):
            continue

        full_path = os.path.join(data_path, fname)
        pdf = pymupdf.open(full_path)
        num_pages = len(pdf)
        print(f"Processing {fname}: {num_pages} pages")

        for page_num in range(num_pages):
            raw_text = pdf[page_num].get_text("text")
            docs.append(
                Document(
                    page_content=raw_text,
                    metadata={
                        'source': fname,
                        'page': page_num + 1
                    }
                )
            )
        pdf.close()

    return splitter.split_documents(docs)


def calculate_chunk_ids(chunks: List[Document]) -> None:
    last: Optional[str] = None
    idx = 0
    for c in chunks:
        key = f"{c.metadata['source']}:{c.metadata['page']}"
        idx = idx + 1 if key == last else 0
        c.metadata['id'] = f"{key}:{idx}"
        last = key


def build_chroma(
    chunks: List[Document],
    persist_directory: str = 'chroma',
    reset: bool = False
) -> Chroma:
    if reset and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=get_embedding_function())
    calculate_chunk_ids(chunks)
    existing = set(db.get().get('ids', []))
    new = [c for c in chunks if c.metadata['id'] not in existing]
    if new:
        db.add_documents(documents=new, ids=[c.metadata['id'] for c in new])
    return db


def query_and_format(
    db: Chroma,
    input_meta: Dict[str, Any],
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    results = db.similarity_search_with_score(query, k=top_k)
    raw = input_meta.get('documents', [])
    docs_list = [os.path.basename(item['filename']) if isinstance(item, dict) else os.path.basename(item)
                 for item in raw]

    out: Dict[str, Any] = {
        'metadata': {
            'input_documents': docs_list,
            'persona': input_meta.get('persona', {}).get('role', ''),
            'job_to_be_done': input_meta.get('job_to_be_done', {}).get('task', '')
        },
        'extracted_sections': [],
        'subsection_analysis': []
    }

    for rank, (doc, _) in enumerate(results, 1):
        lines = [ln.strip() for ln in doc.page_content.splitlines() if ln.strip()]
        title = lines[0] if lines else ''
        out['extracted_sections'].append({
            'document': doc.metadata['source'],
            'section_title': title,
            'rank': rank
        })

    for doc, _ in results:
        snippet = doc.page_content[:200].strip()
        out['subsection_analysis'].append({
            'document': doc.metadata['source'],
            'text': snippet
        })

    return out


def output_analysis(
    query: str,
    input_json_path: str,
    data_path: str = 'data',
    persist_directory: str = 'chroma',
    reset: bool = True
) -> Dict[str, Any]:
    chunks = load_and_split_documents(data_path)
    db = build_chroma(chunks, persist_directory, reset)
    with open(input_json_path, 'r') as f:
        spec = json.load(f)
    result = query_and_format(db, spec, query)

    with open('output.json', 'w') as f_out:
        json.dump(result, f_out, indent=2)
    print(json.dumps(result, indent=2))
    return result


def Output_Analysis(
    query: str,
    input_json_path: str
) -> None:
    _ = output_analysis(query, input_json_path)

# usage:
Output_Analysis('how to play monopoly', 'input_small.json')
