"""
Example Usage of Advanced RAG Pipeline

This script demonstrates different ways to use the pipeline with various configurations.
"""

import os
import sys
from pathlib import Path

# Import the pipeline and configuration
from advanced_rag_pipeline import AdvancedRAGPipeline
from config_rag import (
    LLM_CONFIG, 
    EMBEDDING_CONFIG, 
    RETRIEVAL_CONFIG,
    load_preset,
    get_device
)


def example_basic_usage():
    """Example 1: Basic usage with default settings"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Initialize with default settings
    pipeline = AdvancedRAGPipeline(
        llm_model_name="Qwen/Qwen2.5-3B-Instruct",
        embedding_model_name="BAAI/bge-m3",
        device="cuda:0",
        persist_dir="./storage"
    )
    
    # Ingest documents
    md_files = ["/path/to/doc1.md", "/path/to/doc2.md"]
    # pipeline.ingest_documents(md_files)
    
    # Query
    # answer = pipeline.query("What is a resistor?")
    # print(answer.answer)
    
    print("✓ Basic pipeline initialized successfully")


def example_with_config():
    """Example 2: Using configuration file"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Using Configuration File")
    print("="*80)
    
    # Initialize with config
    pipeline = AdvancedRAGPipeline(
        llm_model_name=LLM_CONFIG["model_name"],
        embedding_model_name=EMBEDDING_CONFIG["model_name"],
        device=get_device(),
        persist_dir="./storage"
    )
    
    print(f"✓ Using LLM: {LLM_CONFIG['model_name']}")
    print(f"✓ Using Embeddings: {EMBEDDING_CONFIG['model_name']}")
    print(f"✓ Device: {get_device()}")


def example_with_preset():
    """Example 3: Using configuration presets"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Using Configuration Presets")
    print("="*80)
    
    # Load 'fast' preset for quick testing
    preset = load_preset("balanced")
    
    # Initialize with preset
    pipeline = AdvancedRAGPipeline(
        llm_model_name=LLM_CONFIG["model_name"],
        embedding_model_name=EMBEDDING_CONFIG["model_name"],
        device=get_device(),
        persist_dir="./storage"
    )
    
    print(f"✓ Using 'balanced' preset")
    print(f"  LLM: {LLM_CONFIG['model_name']}")
    print(f"  Embeddings: {EMBEDDING_CONFIG['model_name']}")
    print(f"  Hybrid Top-K: {RETRIEVAL_CONFIG['hybrid_top_k']}")
    print(f"  Rerank Top-K: {RETRIEVAL_CONFIG['rerank_top_k']}")


def example_custom_models():
    """Example 4: Using custom models"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Using Custom Models")
    print("="*80)
    
    # You can use any HuggingFace model
    pipeline = AdvancedRAGPipeline(
        llm_model_name="microsoft/phi-2",  # Smaller, faster model
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster embeddings
        device="cuda:0",
        persist_dir="./storage_custom"
    )
    
    print("✓ Using custom models:")
    print("  LLM: microsoft/phi-2")
    print("  Embeddings: sentence-transformers/all-MiniLM-L6-v2")


def example_query_variations():
    """Example 5: Different query options"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Query Variations")
    print("="*80)
    
    # Initialize pipeline (would be done once)
    # pipeline = AdvancedRAGPipeline(...)
    
    question = "What are the components of a circuit?"
    
    print("\nQuery options:")
    
    # Standard query
    print("\n1. Standard query:")
    print(f"   answer = pipeline.query('{question}')")
    
    # With HyDE
    print("\n2. Query with HyDE (for vague questions):")
    print(f"   answer = pipeline.query('{question}', use_hyde=True)")
    
    # With decomposition
    print("\n3. Query with decomposition (for complex questions):")
    print(f"   answer = pipeline.query('{question}', use_decomposition=True)")
    
    # Both
    print("\n4. Query with both:")
    print(f"   answer = pipeline.query('{question}', use_hyde=True, use_decomposition=True)")


def example_processing_csv():
    """Example 6: Processing questions from CSV"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Processing Questions from CSV")
    print("="*80)
    
    code = """
import pandas as pd

# Initialize pipeline
pipeline = AdvancedRAGPipeline(...)
pipeline.ingest_documents(md_files)

# Load questions
df = pd.read_csv("questions.csv")

results = []
for idx, row in df.iterrows():
    question = row['Question']
    
    # Query
    answer = pipeline.query(question)
    
    # Extract answer choices (simple heuristic)
    choices = []
    for choice in ['A', 'B', 'C', 'D']:
        if choice in row and pd.notna(row[choice]):
            if str(row[choice]).lower() in answer.answer.lower():
                choices.append(choice)
    
    result = f"{idx + 1},{','.join(choices) if choices else 'A'}"
    results.append(result)

# Save results
with open("answers.txt", 'w') as f:
    f.write('\\n'.join(results))
"""
    print(code)


def example_accessing_citations():
    """Example 7: Accessing and using citations"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Accessing Citations")
    print("="*80)
    
    code = """
# Query the pipeline
answer = pipeline.query("What is a resistor?")

# Access the structured answer
print(f"Answer: {answer.answer}")
print(f"Confidence: {answer.confidence}")

# Iterate through citations
for i, citation in enumerate(answer.citations, 1):
    print(f"\\nCitation {i}:")
    print(f"  Document: {citation.document_name}")
    print(f"  Section: {citation.section_header}")
    print(f"  Snippet: {citation.snippet}")

# Convert to dictionary
answer_dict = {
    "answer": answer.answer,
    "citations": [
        {
            "document": c.document_name,
            "section": c.section_header,
            "snippet": c.snippet
        }
        for c in answer.citations
    ],
    "confidence": answer.confidence
}

# Save as JSON
import json
with open("answer.json", "w") as f:
    json.dump(answer_dict, f, indent=2, ensure_ascii=False)
"""
    print(code)


def example_cpu_usage():
    """Example 8: Running on CPU (no GPU)"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Running on CPU")
    print("="*80)
    
    # For systems without GPU
    pipeline = AdvancedRAGPipeline(
        llm_model_name="microsoft/phi-2",  # Use smaller model for CPU
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",  # Use CPU
        persist_dir="./storage_cpu"
    )
    
    print("✓ Pipeline configured for CPU")
    print("  Note: This will be slower than GPU but will work on any system")


def example_incremental_ingestion():
    """Example 9: Incremental document ingestion"""
    print("\n" + "="*80)
    print("EXAMPLE 9: Incremental Ingestion")
    print("="*80)
    
    code = """
# Initialize pipeline
pipeline = AdvancedRAGPipeline(...)

# Ingest initial batch
batch1 = ["doc1.md", "doc2.md", "doc3.md"]
pipeline.ingest_documents(batch1)

# Later, ingest more documents
batch2 = ["doc4.md", "doc5.md"]
# Note: This will add to existing index
pipeline.ingest_documents(batch2)

# All documents are now searchable
"""
    print(code)
    print("\nNote: Currently, re-ingestion rebuilds the entire index.")
    print("For production, consider implementing incremental updates.")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("ADVANCED RAG PIPELINE - USAGE EXAMPLES")
    print("="*80)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Using Configuration", example_with_config),
        ("Using Presets", example_with_preset),
        ("Custom Models", example_custom_models),
        ("Query Variations", example_query_variations),
        ("Processing CSV", example_processing_csv),
        ("Accessing Citations", example_accessing_citations),
        ("CPU Usage", example_cpu_usage),
        ("Incremental Ingestion", example_incremental_ingestion),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning examples...")
    
    try:
        for name, example_func in examples:
            try:
                example_func()
            except Exception as e:
                print(f"\n✗ Error in {name}: {e}")
                import traceback
                traceback.print_exc()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80)
    
    print("\nNext Steps:")
    print("1. Install dependencies: pip install -r requirements_rag.txt")
    print("2. Prepare your markdown documents")
    print("3. Run test_rag_pipeline.py for a complete test")
    print("4. Or run advanced_rag_pipeline.py with your questions.csv")
    print("5. Customize config_rag.py to change models or parameters")


if __name__ == "__main__":
    main()
