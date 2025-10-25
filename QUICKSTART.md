# Quick Start Guide - Advanced RAG Pipeline

Get started with the Advanced RAG Pipeline in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended, but can run on CPU)
- At least 16GB RAM
- 10GB free disk space for models

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements_rag.txt
```

This will install:
- LlamaIndex and extensions
- Transformers and HuggingFace libraries
- BM25, pandas, numpy, and other utilities

### Step 2: Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Quick Test

### Test with Sample Documents

Run the test script which creates sample documents:

```bash
python test_rag_pipeline.py
```

This will:
1. Create sample markdown documents
2. Initialize the pipeline
3. Ingest documents
4. Run test queries
5. Display results

Expected output:
```
Advanced RAG Pipeline - Test Script
================================================================================
Creating sample documents in /tmp/test_documents
Found 3 markdown files:
  - /tmp/test_documents/electronics_basics.md
  - /tmp/test_documents/presentation_software.md
  - /tmp/test_documents/advanced_electronics.md

================================================================================
Initializing Advanced RAG Pipeline...
================================================================================
Loading embedding model: BAAI/bge-m3
Loading LLM: Qwen/Qwen2.5-3B-Instruct
...
```

## Basic Usage

### 1. Prepare Your Documents

Organize your markdown files in a directory:

```
/path/to/documents/
  ├── doc1.md
  ├── doc2.md
  ├── doc3.md
  └── ...
```

### 2. Create a Simple Script

```python
from advanced_rag_pipeline import AdvancedRAGPipeline
import os

# Get your markdown files
doc_dir = "/path/to/documents"
md_files = [os.path.join(doc_dir, f) for f in os.listdir(doc_dir) if f.endswith('.md')]

# Initialize pipeline
pipeline = AdvancedRAGPipeline(
    llm_model_name="Qwen/Qwen2.5-3B-Instruct",
    embedding_model_name="BAAI/bge-m3",
    device="cuda:0",  # or "cpu" if no GPU
    persist_dir="./storage"
)

# Ingest documents (do this once)
pipeline.ingest_documents(md_files)

# Query
answer = pipeline.query("Your question here?")

# Display results
print(f"Answer: {answer.answer}")
print(f"\nCitations:")
for citation in answer.citations:
    print(f"- {citation.document_name}: {citation.section_header}")
```

### 3. Run Your Script

```bash
python your_script.py
```

## Process Questions from CSV

### Prepare questions.csv

Create a CSV file with questions:

```csv
Question,A,B,C,D
What is a resistor?,A passive component,An active component,A power source,A switch
...
```

### Run the Main Pipeline

```bash
python advanced_rag_pipeline.py
```

This will:
1. Load all markdown files from the configured directory
2. Process each question in questions.csv
3. Generate answers with citations
4. Save results to answers.txt

## Configuration

### Quick Configuration Changes

Edit `config_rag.py` to customize:

```python
# Change models
LLM_CONFIG["model_name"] = "microsoft/phi-2"  # Smaller, faster
EMBEDDING_CONFIG["model_name"] = "BAAI/bge-base-en-v1.5"  # Faster

# Change retrieval parameters
RETRIEVAL_CONFIG["hybrid_top_k"] = 30  # Fewer candidates
RETRIEVAL_CONFIG["rerank_top_k"] = 3   # Top 3 results

# Change device
DEVICE_CONFIG["device"] = "cpu"  # Use CPU instead of GPU
```

### Use Configuration Presets

```python
from config_rag import load_preset

# Load a preset configuration
load_preset("fast")      # Fast but less accurate
# or
load_preset("balanced")  # Good balance (default)
# or  
load_preset("accurate")  # More accurate but slower
```

## Common Issues & Solutions

### Issue: Out of Memory (OOM)

**Solution 1**: Use a smaller model
```python
pipeline = AdvancedRAGPipeline(
    llm_model_name="microsoft/phi-2",  # Only 2.7B parameters
    ...
)
```

**Solution 2**: Use CPU for LLM
```python
pipeline = AdvancedRAGPipeline(
    device="cpu",
    ...
)
```

**Solution 3**: Reduce retrieval size
```python
from config_rag import RETRIEVAL_CONFIG
RETRIEVAL_CONFIG["hybrid_top_k"] = 20  # Instead of 50
RETRIEVAL_CONFIG["rerank_top_k"] = 3   # Instead of 5
```

### Issue: Slow Performance

**Solution 1**: Use fast preset
```python
from config_rag import load_preset
load_preset("fast")
```

**Solution 2**: Disable HyDE and decomposition
```python
answer = pipeline.query(question, use_hyde=False, use_decomposition=False)
```

**Solution 3**: Use smaller models
```python
pipeline = AdvancedRAGPipeline(
    llm_model_name="microsoft/phi-2",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    ...
)
```

### Issue: Models Not Downloading

**Solution 1**: Check internet connection

**Solution 2**: Set HuggingFace cache directory
```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

**Solution 3**: Download manually
```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
huggingface-cli download BAAI/bge-m3
```

### Issue: Import Errors

**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements_rag.txt --upgrade
```

## Performance Tips

### 1. First-Time Setup
- Models will be downloaded on first run (~5-10 GB)
- This can take 10-30 minutes depending on internet speed
- Models are cached for future use

### 2. Ingestion
- Ingest documents once, then reuse the index
- Index is saved to `persist_dir`
- Reload existing index instead of re-ingesting:
  ```python
  # If index exists, it will be loaded automatically
  pipeline = AdvancedRAGPipeline(persist_dir="./storage")
  ```

### 3. Query Optimization
- Use HyDE only for vague queries
- Use decomposition only for complex queries
- For simple queries, disable both for faster results

### 4. Batch Processing
- Process multiple questions in one session
- Avoid reinitializing the pipeline multiple times

## Next Steps

1. **Read the Full Documentation**: See `README_RAG.md` for detailed information

2. **Explore Examples**: Run `python examples_rag.py` to see various usage patterns

3. **Customize Configuration**: Edit `config_rag.py` to tune the pipeline

4. **Integrate with Your Application**: Use the pipeline as a module in your code

5. **Monitor Performance**: Track query times and accuracy for your use case

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review `README_RAG.md` for detailed documentation
3. Check LlamaIndex docs: https://docs.llamaindex.ai/
4. Verify model compatibility on HuggingFace

## File Overview

- `advanced_rag_pipeline.py` - Main pipeline implementation
- `config_rag.py` - Configuration settings
- `test_rag_pipeline.py` - Test script with sample documents
- `examples_rag.py` - Usage examples
- `requirements_rag.txt` - Python dependencies
- `README_RAG.md` - Full documentation
- `QUICKSTART.md` - This file

## Minimum Working Example

The absolute minimum code to get started:

```python
from advanced_rag_pipeline import AdvancedRAGPipeline

# Initialize
pipeline = AdvancedRAGPipeline()

# Ingest
pipeline.ingest_documents(["doc1.md", "doc2.md"])

# Query
answer = pipeline.query("What is X?")
print(answer.answer)
```

That's it! You're ready to build powerful Q&A systems over your documents.
