import os
import re
import json
import fitz  # PyMuPDF
from datetime import datetime
from typing import List, Dict, Any, Optional

# LangChain imports (offline-compatible)
from fastembed import TextEmbedding
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GPT4All
from langchain.chains import RetrievalQA


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"(\*|_|`|#{1,6})+", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_and_split(
    pdf_paths: List[str],
    chunk_size: int = 800,
    chunk_overlap: int = 80
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    docs = []
    for path in pdf_paths:
        name = os.path.basename(path)
        with fitz.open(path) as pdf:
            for i, page in enumerate(pdf, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    chunks = splitter.split_text(text)
                    for chunk in chunks:
                        docs.append(Document(page_content=chunk, metadata={"source": name, "page": i}))
    return docs


class FastEmbedWrapper:
    def __init__(self, model: TextEmbedding):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return list(self.model.embed(texts))

    def embed_query(self, text: str) -> List[float]:
        return list(self.model.embed([text]))[0]


def build_and_persist(
    pdf_dir: str,
    persist_dir: str = "chroma",
    reset: bool = True,
    spec_path: Optional[str] = None
) -> Chroma:
    if os.path.isdir(pdf_dir):
        pdfs = [os.path.join(pdf_dir, f)
                for f in sorted(os.listdir(pdf_dir))
                if f.lower().endswith('.pdf')]
    elif spec_path and os.path.isfile(spec_path):
        spec = json.load(open(spec_path, 'r'))
        docs = spec.get('documents', [])
        pdfs = [os.path.abspath(item['filename'] if isinstance(item, dict) else item)
                for item in docs]
    else:
        raise FileNotFoundError(f"No PDFs found in '{pdf_dir}' and no valid spec provided.")

    if reset and os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)

    embeddings = FastEmbedWrapper(TextEmbedding())
    vectordb = Chroma.from_documents(
        documents=extract_and_split(pdfs),
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb


def make_offline_qa_chain(
    vectordb: Chroma,
    model_path: str = "models/gpt4all-model.bin",
    n_ctx: int = 2048,
    n_threads: int = 4
) -> RetrievalQA:
    llm = GPT4All(
        model=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )


def query_and_format(
    db: Chroma,
    input_meta: Dict[str, Any],
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    results = db.similarity_search_with_score(query, k=top_k)

    docs_list = []
    for item in input_meta.get("documents", []):
        fn = item["filename"] if isinstance(item, dict) else item
        docs_list.append(os.path.basename(fn))

    meta = {
        "input_documents": docs_list,
        "persona": input_meta.get("persona", {}).get("role", ""),
        "job_to_be_done": input_meta.get("job_to_be_done", {}).get("task", ""),
        "processing_timestamp": datetime.utcnow().isoformat()
    }

    extracted = []
    for rank, (doc, _) in enumerate(results, start=1):
        first_line = doc.page_content.splitlines()[0] if doc.page_content else ""
        extracted.append({
            "document": doc.metadata["source"],
            "section_title": _clean_text(first_line),
            "importance_rank": rank,
            "page_number": doc.metadata.get("page", None)
        })

    analysis = []
    for doc, _ in results:
        snippet = _clean_text(doc.page_content)[:300]
        analysis.append({
            "document": doc.metadata["source"],
            "refined_text": snippet,
            "page_number": doc.metadata.get("page", None)
        })

    return {
        "metadata": meta,
        "extracted_sections": extracted,
        "subsection_analysis": analysis
    }


def output_analysis(
    query: str,
    input_json_path: str,
    data_path: str = 'data',
    persist_directory: str = 'chroma',
    reset: bool = True
) -> Dict[str, Any]:
    with open(input_json_path) as f:
        spec = json.load(f)

    db = build_and_persist(
        pdf_dir=data_path,
        persist_dir=persist_directory,
        reset=reset,
        spec_path=input_json_path
    )

    return query_and_format(db, spec, query)


# Run
if __name__ == "__main__":
    result = output_analysis(
        query="How to play Monopoly",
        input_json_path="input_small.json",
        data_path="data",  # or override if directory doesn't exist
        persist_directory="chroma",
        reset=True
    )

    print(json.dumps(result, indent=2))
