#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import shutil
import re
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import argparse
import pymupdf  # PyMuPDF for plain-text extraction
from fastembed import TextEmbedding
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    # Normalize common unicode ligatures → ASCII
    ligatures = {
        '\ufb00': 'ff',  # ﬀ
        '\ufb01': 'fi',  # ﬁ
        '\ufb02': 'fl',  # ﬂ
        '\ufb03': 'ffi', # ﬃ
        '\ufb04': 'ffl', # ﬄ
        '\ufb05': 'ft',  # ﬅ
        '\ufb06': 'st',  # ﬆ
    }
    for uni, ascii_equiv in ligatures.items():
        text = text.replace(uni, ascii_equiv)

    # Convert unicode bullets to a dash
    # Handle both explicit escape and actual char
    text = text.replace('\\u2022', '-')
    text = text.replace('•', '-')
    # Remove redundant bullet-dot combos
    text = text.replace('-.', '-')
    # Replace smart quotes with straight ones
    text = text.replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    # Strip common Markdown syntax
    text = re.sub(r'#+' , '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Collapse multiple spaces
    text = re.sub(r"[ ]{2,}", ' ', text)
    # Replace newlines with spaces so sentences only end at actual periods
    text = re.sub(r"\s*\n+\s*", ' ', text)
    # Ensure sentence boundaries only at periods
    parts = [p.strip() for p in text.split('. ') if p.strip()]
    text = '. '.join(parts)
    text = text.strip(' .')
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    return text


def get_embedding_function(model_path: Optional[str] = None):
    """
    Returns an embedding function wrapper that loads a fastembed model offline.
    If `model_path` is provided (directory or file), fastembed will load from there.
    Otherwise, it uses the default cached model.
    """
    class EmbeddingWrapper:
        def __init__(self):
            if model_path:
                self.model = TextEmbedding(model_name_or_path=model_path)
            else:
                self.model = TextEmbedding()

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return list(self.model.embed(texts))

        def embed_query(self, text: str) -> List[float]:
            return list(self.model.embed([text]))[0]

    return EmbeddingWrapper()


def load_and_split_documents(
    data_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> List[Document]:
    docs: List[Document] = []
    # Sentence-aware splitting on full stops
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=['\\. ', '\n\n', '\n', ' ', '']
    )

    for fname in sorted(os.listdir(data_path)):
        if not fname.lower().endswith('.pdf'):
            continue
        full_path = os.path.join(data_path, fname)
        pdf = pymupdf.open(full_path)
        for page_num in range(len(pdf)):
            raw_text = pdf[page_num].get_text("text")
            docs.append(Document(page_content=raw_text, metadata={'source': fname, 'page': page_num + 1}))
        pdf.close()

    # Split documents into chunks that respect sentence boundaries
    chunks = splitter.split_documents(docs)
    # Trim each chunk to end at the last full stop
    for c in chunks:
        content = c.page_content
        last_dot = content.rfind('.')
        if last_dot != -1:
            c.page_content = content[:last_dot+1]
    return chunks


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
    embedding_function=None
) -> Chroma:
    # Always reset to repopulate database
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    # Use provided embedding function or fall back to global one
    if embedding_function is None:
        embedding_function = globals().get('embedded_fn') or get_embedding_function()
    
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    calculate_chunk_ids(chunks)
    db.add_documents(documents=chunks, ids=[c.metadata['id'] for c in chunks])
    return db


def query_and_format(
    db: Chroma,
    input_meta: Dict[str, Any],
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    results = db.similarity_search_with_score(query, k=top_k)
    raw_docs = input_meta.get('documents', [])
    docs_list = [os.path.basename(item['filename']) if isinstance(item, dict) else os.path.basename(item) for item in raw_docs]
    out: Dict[str, Any] = {
        'metadata': {
            'input_documents': docs_list,
            'persona': input_meta.get('persona', {}).get('role', ''),
            'job_to_be_done': input_meta.get('job_to_be_done', {}).get('task', '')
        },
        'extracted_sections': [],
        'subsection_analysis': []
    }
    for rank, (doc, _) in enumerate(results, start=1):
        lines = [ln.strip() for ln in doc.page_content.splitlines() if ln.strip()]
        title = lines[0] if lines else ''
        out['extracted_sections'].append({'document': doc.metadata['source'], 'section_title': title, 'rank': rank})

    for doc, _ in results:
        # snippet truncated at last full stop
        raw_snippet = doc.page_content[:200]
        last_dot = raw_snippet.rfind('.')
        if last_dot != -1:
            raw_snippet = raw_snippet[: last_dot + 1]
        cleaned = clean_text(raw_snippet)
        out['subsection_analysis'].append({'document': doc.metadata['source'], 'text': cleaned})
    return out


def get_json_result_for_query(
    input_spec: Union[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    if isinstance(input_spec, str):
        input_spec_path = os.path.abspath(input_spec)
        with open(input_spec_path) as f:
            spec = json.load(f)
        spec_dir = os.path.dirname(input_spec_path)
    else:
        spec = input_spec
        spec_dir = os.getcwd()

    embedding_fn = get_embedding_function(model_path)

    if 'query' in spec:
        query = spec.pop('query')
    else:
        challenge = spec.get('challenge_info', {})
        query = challenge.get('description', '').strip()
        if not query:
            raise ValueError("Specification must include a 'query' field or a 'challenge_info.description'.")

    data_path = spec.get('data_path')
    if not data_path:
        common_dir_names = ['data', 'PDFs', 'pdfs', 'documents', 'docs']
        data_path = next((os.path.join(spec_dir, d) for d in common_dir_names if os.path.exists(os.path.join(spec_dir, d))), None)
        if not data_path:
            data_path = os.path.join(spec_dir, 'data')
    else:
        if not os.path.isabs(data_path):
            data_path = os.path.join(spec_dir, data_path)

    persist_dir = spec.get('persist_dir') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma')
    if not os.path.isabs(persist_dir):
        persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), persist_dir)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    chunks = load_and_split_documents(data_path)
    db = build_chroma(chunks, persist_directory=persist_dir, embedding_function=embedding_fn)
    result = query_and_format(db, spec, query)

    if output_path:
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)
        with open(output_path, 'w') as f_out:
            json.dump(result, f_out, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Offline semantic search: specify local embedding model path.'
    )
    parser.add_argument('input', help='Path to input JSON spec')
    parser.add_argument('output', nargs='?', default='outline.json', help='Optional output JSON file')
    parser.add_argument('--model-path', '-m', help='Local path to pretrained embedding model directory')

    args = parser.parse_args()
    model_path = args.model_path or os.getenv('FASTEMBED_EMBEDDING_MODEL')
    
    global embedded_fn
    embedded_fn = get_embedding_function(model_path)

    try:
        input_path = os.path.abspath(args.input)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        output_path = os.path.abspath(args.output) if args.output else None
        res = get_json_result_for_query(input_path, output_path, model_path)
        print(f'Results saved to {output_path}')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
