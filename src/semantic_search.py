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
    text = text.replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = re.sub(r"\s*\n+\s*", '. ', text)
    text = re.sub(r"[ ]{2,}", ' ', text)
    text = text.strip(' .')
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
                # Load from local path to avoid internet
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
        full_path = os.path.join(data_path, fname)
        pdf = pymupdf.open(full_path)
        for page_num in range(len(pdf)):
            raw_text = pdf[page_num].get_text("text")
            docs.append(Document(page_content=raw_text, metadata={'source': fname, 'page': page_num + 1}))
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
) -> Chroma:
    # Always reset to repopulate database
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embedded_fn)
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
        raw = doc.page_content[:200]
        cleaned = clean_text(raw)
        out['subsection_analysis'].append({'document': doc.metadata['source'], 'text': cleaned})
    return out


def get_json_result_for_query(
    input_spec: Union[str, Dict[str, Any]],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    if isinstance(input_spec, str):
        input_spec_path = os.path.abspath(input_spec)
        with open(input_spec_path) as f:
            spec = json.load(f)
        # Get the directory containing the input spec file
        spec_dir = os.path.dirname(input_spec_path)
    else:
        spec = input_spec
        spec_dir = os.getcwd()

    if 'query' in spec:
        query = spec.pop('query')
    else:
        challenge = spec.get('challenge_info', {})
        query = challenge.get('description', '').strip()
        if not query:
            raise ValueError("Specification must include a 'query' field or a 'challenge_info.description'.")

    # Make paths absolute and robust
    data_path = spec.get('data_path')
    if not data_path:
        # Default to 'data' directory relative to the input spec file
        data_path = os.path.join(spec_dir, 'data')
    else:
        # If relative path, make it relative to the spec file directory
        if not os.path.isabs(data_path):
            data_path = os.path.join(spec_dir, data_path)
    
    persist_dir = spec.get('persist_dir')
    if not persist_dir:
        # Default to 'chroma' in the src directory
        persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma')
    else:
        # If relative path, make it relative to the script directory
        if not os.path.isabs(persist_dir):
            persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), persist_dir)
    
    # Verify data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    chunks = load_and_split_documents(data_path)
    db = build_chroma(chunks, persist_directory=persist_dir)
    result = query_and_format(db, spec, query)

    if output_path:
        # Make output path absolute if it's relative
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
    global embedded_fn
    embedded_fn = get_embedding_function(args.model_path or os.getenv('FASTEMBED_EMBEDDING_MODEL'))

    try:
        # Convert input path to absolute
        input_path = os.path.abspath(args.input)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Convert output path to absolute if provided
        output_path = os.path.abspath(args.output) if args.output else None
        
        res = get_json_result_for_query(input_path, output_path)
        print(f'Results saved to {output_path}')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
