# v6 All cases 
import re
import json
import pymupdf4llm

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

# Example usage:
# print(json.dumps(extract_outline_from_pdf(file_path), indent=2))
