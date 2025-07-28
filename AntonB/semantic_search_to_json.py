"""
Library for building a Chroma vectorstore from PDFs and performing semantic searches,
then formatting results as structured, cleaned JSON (stripped of markdown).
"""
import json
import os
import shutil
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastembed import TextEmbedding
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _clean_text(text: str) -> str:
    """
    Strip markdown punctuation (*, _, `, #), collapse whitespace.
    """
    # remove markdown symbols
    no_md = re.sub(r"(\*|_|`|#{1,6})+", "", text)
    # collapse whitespace
    collapsed = re.sub(r"\s+", " ", no_md)
    return collapsed.strip()


def get_embedding_function():
    return TextEmbedding()


def load_and_split_documents(
    data_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 80
) -> List[Document]:
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
        full = os.path.join(data_path, fname)
        md_pages = __import__('pymupdf4llm').to_markdown(full, page_chunks=True)
        for i, page in enumerate(md_pages, start=1):
            docs.append(
                Document(
                    page_content=page['text'],
                    metadata={'source': os.path.basename(fname), 'page': i}
                )
            )
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
    out = {
        'metadata': {
            'input_documents': docs_list,
            'persona': input_meta.get('persona', {}).get('role', ''),
            'job_to_be_done': input_meta.get('job_to_be_done', {}).get('task', '')
        },
        'extracted_sections': [],
        'subsection_analysis': []
    }
    for rank, (doc, _) in enumerate(results, 1):
        first_line = doc.page_content.splitlines()[0] if doc.page_content else ''
        clean_title = _clean_text(first_line)
        out['extracted_sections'].append({
            'document': doc.metadata['source'],
            'section_title': clean_title,
            'rank': rank
        })
    for doc, _ in results:
        full_text = doc.page_content or ''
        clean_snippet = _clean_text(full_text)[:200]
        out['subsection_analysis'].append({
            'document': doc.metadata['source'],
            'text': clean_snippet
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
    spec = json.load(open(input_json_path))
    result = query_and_format(db, spec, query)
    print(json.dumps(result, indent=2))
    with open('output.json', 'w') as f:
        json.dump(result, f, indent=2)
    return result


def Output_Analysis(
    query: str,
    input_json_path: str
) -> None:
    result = output_analysis(query, input_json_path)

# usage example:
result = output_analysis('create fillable forms', 'input.json')
print(result)
