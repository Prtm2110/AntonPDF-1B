import argparse
from langchain_chroma import Chroma

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Build the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    print(f"Context:\n{context_text}\n\n---\n\n")
    return context_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)

if __name__ == "__main__":
    main()
