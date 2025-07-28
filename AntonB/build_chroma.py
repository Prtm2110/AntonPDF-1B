import argparse
import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pymupdf4llm
from fastembed.embedding import DefaultEmbedding
from llm4_to_json import extract_outline_and_title
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma

# Constants
CHROMA_PATH = 'chroma'
SNIPPET_RADIUS = 100  # characters around title match
DEFAULT_SNIPPET_FALLBACK = 200  # fallback snippet length
MIN_TITLE_LENGTH = 10  # minimum heading length for H1


def clean_text(text: str) -> str:
    """
    Remove control chars, markdown artifacts, tables, and normalize whitespace.
    """
    # Remove control characters
    text = re.sub(r"[\x00-\x1F]+", " ", text)
    # Remove markdown symbols and brackets
    text = re.sub(r"[#>*`_\[\]\(\)]+", "", text)
    # Remove markdown table lines
    text = re.sub(r"(?m)^\s*\|.*\|.*$", "", text)
    # Collapse spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize newlines
    text = re.sub(r"\s*\n+\s*", "\n", text)
    return text.strip()


class EmbeddingWrapper:
    """Wrap DefaultEmbedding to satisfy Chroma's API."""
    def __init__(self):
        self.embedder = DefaultEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="emb_models"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            vecs = list(self.embedder.embed(text))
            embeddings.append(vecs[0] if vecs else [0.0] * self.embedder.embedding_size)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def parse_pdf(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert PDF to cleaned text and extract outline entries.
    Returns:
        cleaned full text, list of outline dicts
    """
    md_pages = pymupdf4llm.to_markdown(path, page_chunks=True)
    outline_data = extract_outline_and_title(md_pages)
    outline = outline_data.get('outline', [])
    raw_text = ''.join(page.get('text', '') for page in md_pages)
    cleaned = clean_text(raw_text)
    return cleaned, outline


def load_input(path: str) -> Dict[str, Any]:
    """Load and validate the input JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Input JSON root must be an object, got {type(data)}")
    return data


def build_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the metadata section."""
    raw_docs = data.get('documents', [])
    filenames: List[str] = []
    for item in raw_docs:
        if isinstance(item, dict) and 'filename' in item:
            filenames.append(item['filename'])
        elif isinstance(item, str):
            filenames.append(item)
    persona = data.get('persona', {}).get('role', '')
    job = data.get('job_to_be_done', {}).get('task', '')
    return {
        'input_documents': filenames,
        'persona': persona,
        'job_to_be_done': job,
        'processing_timestamp': datetime.utcnow().isoformat()
    }


def extract_sections(doc_outlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select the top 5 H1 headings across all documents."""
    h1_list: List[Dict[str, Any]] = []
    for doc in doc_outlines:
        filename = doc.get('filename', '')
        for entry in doc.get('outline', []):
            if entry.get('level') == 'H1':
                title = clean_text(entry.get('text', ''))
                if len(title) >= MIN_TITLE_LENGTH:
                    h1_list.append({
                        'document': filename,
                        'section_title': title,
                        'page_number': entry.get('page', 0)
                    })
    top5 = h1_list[:5]
    for idx, sec in enumerate(top5, start=1):
        sec['importance_rank'] = idx
    return top5


def refine_subsections(
    text_meta: List[Dict[str, Any]],
    sections: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract a context snippet around each section title."""
    results: List[Dict[str, Any]] = []
    for sec in sections:
        doc = sec['document']
        title_frag = re.escape(sec['section_title'][:30])
        pattern = re.compile(
            rf"(.{{0,{SNIPPET_RADIUS}}}){title_frag}(.{{0,{SNIPPET_RADIUS}}})",
            re.IGNORECASE
        )
        full_text = ''
        for item in text_meta:
            if item['document'] == doc:
                full_text = item['text']
                break
        match = pattern.search(full_text)
        if match:
            snippet = clean_text(match.group(1) + match.group(0) + match.group(2))
        else:
            snippet = clean_text(full_text[:DEFAULT_SNIPPET_FALLBACK])
        results.append({
            'document': doc,
            'refined_text': snippet,
            'page_number': sec['page_number']
        })
    return results


def clear_chroma():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🗑️ Removed previous Chroma vectorstore")


def index_to_chroma(text_meta: List[Dict[str, Any]]):
    """Index text using Chroma and our embedding wrapper."""
    texts = [item['text'] for item in text_meta]
    sources = [item['document'] for item in text_meta]
    ids = [f"{i}-{os.path.basename(src)}" for i, src in enumerate(sources)]
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=EmbeddingWrapper()
    )
    docs = [Document(page_content=text, metadata={'source': src})
            for text, src in zip(texts, sources)]
    db.add_documents(docs, ids=ids)
    # db.persist()
    print(f"💾 Indexed {len(docs)} documents into Chroma.")


def main():
    parser = argparse.ArgumentParser(
        description="Process PDFs into JSON output and optional Chroma index"
    )
    parser.add_argument('input_json', help='Path to input JSON file')
    parser.add_argument(
        '--data-path', default='.', help='Directory where PDF files reside'
    )
    parser.add_argument(
        '--reset', action='store_true', help='Clear existing Chroma data'
    )
    parser.add_argument(
        '--index', action='store_true', help='Index text into Chroma'
    )
    args = parser.parse_args()

    data = load_input(args.input_json)
    metadata = build_metadata(data)

    if args.reset:
        clear_chroma()

    text_meta: List[Dict[str, Any]] = []
    outlines: List[Dict[str, Any]] = []
    for entry in data.get('documents', []):
        filename = entry.get('filename') if isinstance(entry, dict) else entry
        pdf_path = os.path.join(args.data_path, filename)
        text, outline = parse_pdf(pdf_path)
        text_meta.append({'document': filename, 'text': text})
        outlines.append({'filename': filename, 'outline': outline})

    sections = extract_sections(outlines)
    analysis = refine_subsections(text_meta, sections)

    output = {
        'metadata': metadata,
        'extracted_sections': sections,
        'subsection_analysis': analysis
    }
    print(json.dumps(output, indent=4, ensure_ascii=False))

    if args.index:
        index_to_chroma(text_meta)


if __name__ == '__main__':
    main()
