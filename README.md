# Anton1B - PDF Smart Semantic Search & Query Response

An intelligent PDF processing system that extracts structured information from PDF collections and provides semantic search capabilities for document analysis. Built on the outline PDF extractor from Anton1A, Anton1B delivers exceptional performance when processing large numbers of files.

## Overview

Anton1B is designed for high-speed processing with a strict 60-second constraint for multiple PDFs. The system prioritizes speed while maximizing context capture through:

- **No LLM calls during processing** - Ensures fast execution
- **Advanced semantic search** - Provides superior output quality
- **Optimized post-processing** - Delivers accurate and fast results

The system leverages `pymupdf4llm` to convert PDFs to markdown format with page numbers, using a script from Anton1A to extract titles and outlines with extreme accuracy. 

**Try Anton1A here:** https://github.com/Prtm2110/AntonPDF-1A.git

The extracted markdown text is simultaneously converted and embedded into a vector database, achieving two operations in one computation cycle for maximum efficiency.

Anton1B offers flexible interaction through Python API and CLI tools with multiple configuration options. Examples are provided below. 

## Installation

```bash
git clone https://github.com/Prtm2110/AntonPDF-1B.git
cd Anton1B
pip install -r requirements.txt
```

## Key Features

- **High-Speed Batch Processing**: Process multiple PDF collections simultaneously with sub-60-second performance
- **Built on Anton1A Foundation**: Leverages proven outline PDF extractor optimized for large file volumes
- **Intelligent Content Extraction**: Extracts titles, outlines, and hierarchical structure with high accuracy
- **Advanced Semantic Search**: Vector-based document search and contextual query capabilities
- **Structured JSON Output**: Clean, structured data output for seamless integration
- **Flexible Configuration**: Customizable input/output directories and processing options
- **Zero LLM Overhead**: Processing pipeline avoids LLM calls for maximum speed

## Project Structure

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

## Usage Guide

### PDF Processing with LLM4

The `llm4_to_json.py` script processes PDF collections and extracts structured information using advanced language models. Built on Anton1A's fast outline extractor, it handles large document collections with exceptional performance.

#### Process All Collections

```bash
python llm4_to_json.py --all
```

**What this command does:**
- Scans all `challenge1b_input.json` files from each Collection folder
- Processes all PDFs in each collection's PDFs directory with optimized speed
- Extracts document outlines, titles, and hierarchical structure efficiently
- Saves results to `challenge_outputs_json/` with descriptive filenames
- Provides real-time progress feedback and comprehensive error handling

#### Process a Single Collection

```bash
python llm4_to_json.py --collection "Collection 1"
```

*Perfect for testing or processing specific document sets.*

#### Custom Directory Processing

```bash
python llm4_to_json.py --all --input-dir custom_pdfs --output-dir custom_output
```

*Configure custom directory structures for maximum flexibility.*

#### View All Available Options

```bash
python llm4_to_json.py --help
```

## Output Format

Generated JSON files contain structured data with comprehensive metadata:

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

## Input Configuration

Each collection requires a `challenge1b_input.json` configuration file. See `example_input.json` for template:

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

## Advanced Semantic Search

For advanced semantic search capabilities using vector embeddings and ChromaDB:




### Process Individual Collections

```bash
# Collection 1 - Travel Planning (South of France)
python3 src/semantic_search.py "challenge_pdfs/Collection 1/challenge1b_input.json" "challenge_outputs_json/1stchallenge1b_output_test.json"

# Collection 2 - Adobe Acrobat Learning Materials  
python3 src/semantic_search.py "challenge_pdfs/Collection 2/challenge1b_input.json" "challenge_outputs_json/2ndchallenge1b_output_test.json"

# Collection 3 - Recipe and Meal Planning
python3 src/semantic_search.py "challenge_pdfs/Collection 3/challenge1b_input.json" "challenge_outputs_json/3rdchallenge1b_output_test.json"
```

### Semantic Search Capabilities

- **Vector Embeddings**: Creates semantic representations of document content using state-of-the-art models
- **Similarity Search**: Finds contextually relevant information across document collections
- **ChromaDB Integration**: Efficient vector storage and lightning-fast retrieval
- **Query-based Results**: Returns most relevant sections based on semantic meaning rather than keyword matching

## Dependencies

### Core Libraries

- **`pymupdf4llm`** - High-performance PDF text extraction and processing
- **`fastembed`** - Fast, efficient embedding model generation
- **`langchain-chroma`** - Vector database for semantic search operations
- **`langchain-community`** - Community tools and integrations
- **`langchain-text-splitters`** - Intelligent text chunking and splitting

*Install all dependencies with: `pip install -r requirements.txt`*

## Technical Architecture

Anton1B implements a sophisticated multi-stage processing pipeline built on Anton1A's proven foundation:

### Processing Pipeline

1. **PDF Text Extraction**
   - Converts PDFs to markdown using PyMuPDF4LLM with optimized performance
   - Preserves page numbers and structural information

2. **Content Parsing**
   - Extracts hierarchical structure (headings, sections) at high speed
   - Maintains document organization and relationships

3. **Text Normalization**
   - Handles Unicode, formatting, and punctuation cleanup efficiently
   - Ensures consistent text processing across documents

4. **Structure Analysis**
   - Identifies document outlines and importance rankings rapidly
   - Ranks content by relevance and significance

5. **JSON Serialization**
   - Outputs structured data for downstream processing
   - Maintains data integrity and accessibility

### Performance Optimizations

- **Zero LLM Calls**: Eliminates API latency for maximum speed
- **Vectorized Operations**: Leverages NumPy and optimized libraries
- **Concurrent Processing**: Handles multiple documents simultaneously
- **Memory Efficient**: Streams data to minimize memory footprint

## Contributing

We welcome contributions to Anton1B! Here's how to get started:

1. **Fork the repository** on GitHub
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with clear, documented code
4. **Commit your changes** (`git commit -m 'Add amazing feature'`)
5. **Push to the branch** (`git push origin feature/amazing-feature`)
6. **Open a Pull Request** with a clear description of your changes

## License

This project is part of the Anton1B challenge system. Please refer to the repository for specific licensing terms.

### Getting Help

- **Check Issues**: Browse existing GitHub issues for solutions
- **Report Bugs**: Open a new issue with detailed error messages and system information
- **Feature Requests**: Propose new features through GitHub issues
- **Documentation**: Refer to inline code comments and docstrings
