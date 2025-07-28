# 📚 PDF Smart Semantic Search & Query Response

An intelligent PDF processing system that extracts structured information from PDF collections and provides semantic search capabilities for document analysis.

## 🚀 Quick Start

```bash
git clone https://github.com/Prtm2110/AntonPDF-1B.git
cd Anton1B
pip install -r requirements.txt
```

## 🔧 Features

- **Batch PDF Processing**: Process multiple PDF collections simultaneously
- **Intelligent Content Extraction**: Extract titles, outlines, and structured data
- **Semantic Search**: Advanced document search and query capabilities
- **JSON Output**: Structured data output for easy integration
- **Flexible Configuration**: Customizable input/output directories

## 📁 Project Structure

```
Anton1B/
├── llm4_to_json.py          # Main PDF processing script
├── process_all.sh           # Interactive script for PDF processing
├── semantic_search_all.sh   # Batch semantic search processing
├── example_input.json       # Template for input configuration
├── src/
│   └── semantic_search.py   # Semantic search functionality
├── challenge_pdfs/          # Input PDF collections
│   ├── Collection 1/
│   ├── Collection 2/
│   └── Collection 3/
├── challenge_outputs_json/  # Generated output files
└── requirements.txt         # Python dependencies
```

## 🛠️ PDF Processing with LLM4

The `llm4_to_json.py` script processes PDF collections and extracts structured information using advanced language models.

### Process All Collections at Once

```bash
python llm4_to_json.py --all
```

**What this does:**
- 📖 Reads all `challenge1b_input.json` files from each Collection folder
- 🔍 Processes all PDFs in each collection's PDFs directory
- 📊 Extracts document outlines, titles, and hierarchical structure
- 💾 Saves results to `challenge_outputs_json/` with descriptive filenames
- ⚡ Provides progress feedback and error handling

### Process a Single Collection

```bash
python llm4_to_json.py --collection "Collection 1"
```

Perfect for testing or processing specific document sets.

### Custom Directories

```bash
python llm4_to_json.py --all --input-dir custom_pdfs --output-dir custom_output
```

Use your own directory structure for maximum flexibility.

### View All Options

```bash
python llm4_to_json.py --help
```

## 📊 Output Format

Generated JSON files contain:

```json
{
  "metadata": {
    "input_documents": ["document1.pdf", "document2.pdf"],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip for college friends",
    "processing_timestamp": "2025-07-28T23:13:36.489538"
  },
  "extracted_sections": [
    {
      "document": "document1.pdf",
      "section_title": "Main Topic",
      "importance_rank": 1,
      "page_number": 1
    }
  ]
}
```

## 📝 Input Configuration

Each collection requires a `challenge1b_input.json` file (see `example_input.json` for template):

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_002",
    "test_case_name": "travel_planner",
    "description": "France Travel"
  },
  "documents": [
    {
      "filename": "document.pdf",
      "title": "Document Title"
    }
  ],
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a trip of 4 days for a group of 10 college friends."
  }
}
```

## 🔍 Semantic Search (Legacy)

For advanced semantic search capabilities using vector embeddings and ChromaDB:


### Manual Processing - All Collections

```bash
# Run all three collections sequentially
python3 src/semantic_search.py "challenge_pdfs/Collection 1/challenge1b_input.json" "challenge_outputs_json/1stchallenge1b_output_test.json" && \
python3 src/semantic_search.py "challenge_pdfs/Collection 2/challenge1b_input.json" "challenge_outputs_json/2ndchallenge1b_output_test.json" && \
python3 src/semantic_search.py "challenge_pdfs/Collection 3/challenge1b_input.json" "challenge_outputs_json/3rdchallenge1b_output_test.json"
```

### Process Individual Collections

```bash
# Collection 1 - Travel Planning (South of France)
python3 src/semantic_search.py "challenge_pdfs/Collection 1/challenge1b_input.json" "challenge_outputs_json/1stchallenge1b_output_test.json"

# Collection 2 - Adobe Acrobat Learning Materials  
python3 src/semantic_search.py "challenge_pdfs/Collection 2/challenge1b_input.json" "challenge_outputs_json/2ndchallenge1b_output_test.json"

# Collection 3 - Recipe and Meal Planning
python3 src/semantic_search.py "challenge_pdfs/Collection 3/challenge1b_input.json" "challenge_outputs_json/3rdchallenge1b_output_test.json"
```

### What Semantic Search Does

- 🧠 **Vector Embeddings**: Creates semantic representations of document content
- 🔍 **Similarity Search**: Finds contextually relevant information across documents
- 📊 **ChromaDB Integration**: Efficient vector storage and retrieval
- 🎯 **Query-based Results**: Returns most relevant sections based on semantic meaning

## 📦 Dependencies

- `pymupdf4llm` - PDF text extraction and processing
- `fastembed` - Fast embedding generation
- `langchain-chroma` - Vector database for semantic search
- `langchain-community` - Community tools and integrations
- `langchain-text-splitters` - Text chunking and splitting

## ⚙️ Technical Approach

The system uses a multi-stage processing pipeline:

1. **PDF Text Extraction**: Converts PDFs to markdown using PyMuPDF4LLM
2. **Content Parsing**: Extracts hierarchical structure (headings, sections)
3. **Text Normalization**: Handles unicode, formatting, and punctuation cleanup
4. **Structure Analysis**: Identifies document outlines and importance rankings
5. **JSON Serialization**: Outputs structured data for downstream processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the AntornPDF-1B challenge system.

## 🐛 Troubleshooting

**Common Issues:**

- **Missing PDFs**: Ensure PDF files exist in the `PDFs/` subdirectory of each collection
- **Permission Errors**: Check write permissions for the output directory
- **Import Errors**: Run `pip install -r requirements.txt` to install dependencies

**Need Help?** Open an issue on GitHub with detailed error messages and system information.


