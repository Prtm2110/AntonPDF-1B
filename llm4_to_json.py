# v6 All cases 
import re
import json
import pymupdf4llm
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

def normalize_punctuation(s: str) -> str:
    """
    Convert common unicode punctuation to ASCII equivalents.
    """
    replacements = {
        "‘": "'",  # left single quote
        "’": "'",  # right single quote
        "“": '"',  # left double quote
        "”": '"',  # right double quote
        "–": '-',  # en dash
        "—": '-',  # em dash
        "…": '...',  # ellipsis
    }
    for uni, ascii_rep in replacements.items():
        s = s.replace(uni, ascii_rep)
    return s


def strip_inline_bold(s: str) -> str:
    """
    Remove backticks, bold markers (**...** or _**...**_),
    trailing numeric markers, repeated punctuation runs,
    collapse multiple spaces, and strip.
    """
    s = normalize_punctuation(s)
    s = re.sub(r'`([^`]+)`', r"\1", s)  # Remove backtick code markers
    s = s.replace('`', '')                # Remove any leftover backticks
    s = re.sub(r'_?\*\*(.*?)\*\*_?', r"\1", s)  # Remove bold markers
    s = re.sub(r'(?:\b)(\d+)$', '', s)              # Remove trailing numeric markers
    s = re.sub(r'[\.\-\,\"\=]{2,}', '', s)      # Remove runs of punctuation
    return ' '.join(s.split()).strip()                # Normalize whitespace


def parse_markdown_outline(md_text: str, page: int):
    outline = []
    for line in md_text.splitlines():
        if re.fullmatch(r"^[\.\-\*,=\"']{2,}\s*$", line.strip()):
            continue  # Skip separators
        stripped = normalize_punctuation(line.strip())

        level = None
        text = None
        if stripped.startswith('# '):
            level = 'H1'
            text = strip_inline_bold(stripped[2:].strip())
        elif stripped.startswith('## '):
            level = 'H1'
            text = strip_inline_bold(stripped[3:].strip())
        elif stripped.startswith('### '):
            level = 'H2'
            text = strip_inline_bold(stripped[4:].strip())
        elif stripped.startswith('#### '):
            level = 'H3'
            text = strip_inline_bold(stripped[5:].strip())
        elif re.fullmatch(r'_?\*\*(.*?)\*\*_?', stripped):
            level = 'H3'
            text = strip_inline_bold(stripped)

        # Filter out anything starting with lowercase after cleanup
        if text and re.match(r'^[a-z]', text):
            continue

        if level and text:
            outline.append({
                'level': level,
                'text': text,
                'page': page + 1
            })
    return outline


def extract_outline_and_title(md_text):
    title = 'Untitled'
    # Skip lowercase lines when finding title
    for page in md_text:
        for line in page.get('text', '').splitlines():
            stripped = normalize_punctuation(line.strip())
            if stripped.startswith('# '):
                candidate = strip_inline_bold(stripped[2:].strip())
                if not re.match(r'^[a-z]', candidate):
                    title = candidate
                    break
        if title != 'Untitled':
            break

    result = {'title': title, 'outline': []}
    for i, page in enumerate(md_text):
        items = parse_markdown_outline(page.get('text', ''), i)
        # Include additional TOC items
        for item in page.get('toc_items', []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lvl, txt = item[0], strip_inline_bold(item[1])
                if not re.match(r'^[a-z]', txt) and txt not in {e['text'] for e in items}:
                    level = f'H{lvl if 1 <= lvl <= 3 else 3}'
                    items.append({'level': level, 'text': txt, 'page': i + 1})
        result['outline'].extend(items)
    return result


def extract_outline_from_pdf(file_path):
    md_text = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    return extract_outline_and_title(md_text)


def process_collection(collection_path, output_dir):
    """Process a single collection directory."""
    collection_name = os.path.basename(collection_path)
    input_file = os.path.join(collection_path, "challenge1b_input.json")
    
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping {collection_name}")
        return None
    
    try:
        # Read input configuration
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Process each PDF in the collection
        pdfs_dir = os.path.join(collection_path, "PDFs")
        if not os.path.exists(pdfs_dir):
            print(f"Warning: PDFs directory not found in {collection_name}")
            return None
        
        extracted_sections = []
        input_documents = []
        
        for doc in input_data.get("documents", []):
            filename = doc["filename"]
            pdf_path = os.path.join(pdfs_dir, filename)
            
            if os.path.exists(pdf_path):
                print(f"Processing {filename}...")
                input_documents.append(filename)
                
                # Extract outline from PDF
                outline_data = extract_outline_from_pdf(pdf_path)
                
                # Convert outline to extracted_sections format
                for i, item in enumerate(outline_data["outline"]):
                    extracted_sections.append({
                        "document": filename,
                        "section_title": item["text"],
                        "importance_rank": i + 1,
                        "page_number": item["page"]
                    })
            else:
                print(f"Warning: {pdf_path} not found")
        
        # Create output structure
        output_data = {
            "metadata": {
                "input_documents": input_documents,
                "persona": input_data.get("persona", {}).get("role", "Unknown"),
                "job_to_be_done": input_data.get("job_to_be_done", {}).get("task", "Unknown"),
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections
        }
        
        # Save output
        output_filename = f"{collection_name.lower().replace(' ', '')}_output.json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Successfully processed {collection_name} -> {output_filename}")
        return output_path
        
    except Exception as e:
        print(f"Error processing {collection_name}: {str(e)}")
        return None


def process_all_collections(challenge_pdfs_dir, output_dir):
    """Process all collections in the challenge_pdfs directory."""
    if not os.path.exists(challenge_pdfs_dir):
        print(f"Error: {challenge_pdfs_dir} not found")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    failed_count = 0
    
    # Find all collection directories
    for item in os.listdir(challenge_pdfs_dir):
        item_path = os.path.join(challenge_pdfs_dir, item)
        
        if os.path.isdir(item_path) and item.startswith("Collection"):
            result = process_collection(item_path, output_dir)
            if result:
                processed_count += 1
            else:
                failed_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {processed_count} collections")
    print(f"  Failed: {failed_count} collections")


def main():
    parser = argparse.ArgumentParser(description="Process PDF collections for challenge 1B")
    parser.add_argument("--all", action="store_true", 
                       help="Process all collections in challenge_pdfs directory")
    parser.add_argument("--collection", type=str,
                       help="Process a specific collection (e.g., 'Collection 1')")
    parser.add_argument("--output-dir", type=str, default="../challenge_outputs_json",
                       help="Output directory for results (default: ../challenge_outputs_json)")
    parser.add_argument("--input-dir", type=str, default="challenge_pdfs",
                       help="Input directory containing collections (default: challenge_pdfs)")
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input_dir)
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    if args.all:
        print(f"Processing all collections from {input_dir}")
        print(f"Output will be saved to {output_dir}")
        process_all_collections(input_dir, output_dir)
    elif args.collection:
        collection_path = os.path.join(input_dir, args.collection)
        if os.path.exists(collection_path):
            print(f"Processing {args.collection}")
            result = process_collection(collection_path, output_dir)
            if result:
                print(f"Success: Output saved to {result}")
            else:
                print("Processing failed")
        else:
            print(f"Error: Collection '{args.collection}' not found in {input_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# Example usage:
# python llm4_to_json.py --all
# python llm4_to_json.py --collection "Collection 1"
