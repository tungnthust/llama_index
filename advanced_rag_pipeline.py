"""
Advanced RAG Pipeline for Document Q&A with Citations

This module implements a sophisticated RAG pipeline with:
- Parent-Child chunking from markdown files with metadata
- Hybrid Search (BM25 + BGE embeddings)
- ColBERT Reranking
- Query Preprocessing (Decomposition and HyDE)
- Structured JSON output with citations
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import torch

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np

# For BM25
from rank_bm25 import BM25Okapi


@dataclass
class Citation:
    """Citation information for a source"""
    document_name: str
    section_header: str
    snippet: str


@dataclass
class StructuredAnswer:
    """Structured answer with citations"""
    answer: str
    citations: List[Citation]
    confidence: Optional[str] = None


class MarkdownParserWithMetadata:
    """Parse markdown files into parent-child chunks with header metadata"""
    
    def __init__(self):
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def parse_markdown_file(self, filepath: str) -> Tuple[List[TextNode], List[TextNode]]:
        """
        Parse a markdown file into parent chunks (sections) and child chunks (paragraphs)
        
        Returns:
            Tuple of (parent_nodes, child_nodes)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        document_name = os.path.basename(filepath)
        
        # Split by headers
        sections = []
        current_section = {'level': 0, 'header': document_name, 'content': '', 'start_pos': 0}
        header_stack = [document_name]
        
        lines = content.split('\n')
        current_content = []
        
        for i, line in enumerate(lines):
            header_match = self.header_pattern.match(line)
            
            if header_match:
                # Save previous section
                if current_content:
                    current_section['content'] = '\n'.join(current_content)
                    sections.append(current_section.copy())
                
                # Start new section
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                
                # Update header stack
                if level <= len(header_stack):
                    header_stack = header_stack[:level]
                header_stack.append(header_text)
                
                current_section = {
                    'level': level,
                    'header': header_text,
                    'content': '',
                    'header_path': ' > '.join(header_stack[1:]),  # Exclude document name
                    'start_pos': i
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            current_section['content'] = '\n'.join(current_content)
            sections.append(current_section)
        
        # Create parent and child nodes
        parent_nodes = []
        child_nodes = []
        
        for idx, section in enumerate(sections):
            if not section['content'].strip():
                continue
            
            # Create parent node (full section)
            parent_id = f"{document_name}_section_{idx}"
            parent_node = TextNode(
                text=section['content'],
                id_=parent_id,
                metadata={
                    'document_name': document_name,
                    'section_header': section['header'],
                    'header_path': section.get('header_path', section['header']),
                    'section_level': section['level'],
                    'node_type': 'parent'
                }
            )
            parent_nodes.append(parent_node)
            
            # Split section into paragraphs for child nodes
            paragraphs = [p.strip() for p in section['content'].split('\n\n') if p.strip()]
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph) < 20:  # Skip very short paragraphs
                    continue
                
                child_id = f"{parent_id}_para_{para_idx}"
                child_node = TextNode(
                    text=paragraph,
                    id_=child_id,
                    metadata={
                        'document_name': document_name,
                        'section_header': section['header'],
                        'header_path': section.get('header_path', section['header']),
                        'section_level': section['level'],
                        'node_type': 'child',
                        'parent_id': parent_id
                    },
                    relationships={
                        NodeRelationship.PARENT: RelatedNodeInfo(node_id=parent_id)
                    }
                )
                child_nodes.append(child_node)
        
        return parent_nodes, child_nodes


class BM25Retriever:
    """BM25 retriever for keyword-based search"""
    
    def __init__(self, nodes: List[TextNode]):
        self.nodes = nodes
        self.corpus = [node.get_content() for node in nodes]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[TextNode, float]]:
        """Retrieve top-k nodes using BM25"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self.nodes[idx], float(scores[idx])))
        
        return results


class ColBERTReranker:
    """ColBERT-based reranker for fine-grained relevance scoring"""
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", device: str = "cuda:0"):
        self.device = device
        self.model_name = model_name
        
        # For simplicity, we'll use a cross-encoder model as a proxy
        # In production, you might want to use actual ColBERT implementation
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    
    def rerank(self, query: str, nodes_with_scores: List[Tuple[TextNode, float]], 
               top_k: int = 5) -> List[Tuple[TextNode, float]]:
        """Rerank nodes using ColBERT/CrossEncoder"""
        if not nodes_with_scores:
            return []
        
        nodes = [item[0] for item in nodes_with_scores]
        texts = [node.get_content() for node in nodes]
        
        # Prepare pairs for cross-encoder
        pairs = [[query, text] for text in texts]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Combine with nodes and sort
        reranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]


class QueryPreprocessor:
    """Preprocess queries with decomposition and HyDE"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        prompt = f"""Given the following question, break it down into simpler sub-questions if it's complex. 
If the question is already simple, return it as is.

Question: {query}

Sub-questions (one per line):"""
        
        response = self.llm.complete(prompt)
        
        # Parse sub-queries
        sub_queries = [q.strip() for q in str(response).split('\n') if q.strip() and not q.strip().startswith('Sub-questions')]
        
        # If no sub-queries found, return original
        if not sub_queries:
            return [query]
        
        return sub_queries
    
    def generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document for HyDE"""
        prompt = f"""Generate a detailed answer to the following question as if you were writing documentation:

Question: {query}

Answer:"""
        
        response = self.llm.complete(prompt)
        return str(response)


class AdvancedRAGPipeline:
    """
    Advanced RAG Pipeline with:
    - Parent-Child chunking
    - Hybrid search (BM25 + Vector)
    - ColBERT reranking
    - Query preprocessing
    - Structured output with citations
    """
    
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = "cuda:0",
        persist_dir: str = "./storage"
    ):
        self.device = device
        self.persist_dir = persist_dir
        
        # Setup device
        torch.cuda.set_device(0)
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            device=device,
            trust_remote_code=True
        )
        
        # Initialize LLM with quantization for efficiency
        print(f"Loading LLM: {llm_model_name}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = HuggingFaceLLM(
            model_name=llm_model_name,
            tokenizer_name=llm_model_name,
            device_map={"": 0},
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": quantization_config,
                "trust_remote_code": True
            },
            generate_kwargs={
                "temperature": 0.1,
                "top_p": 0.95,
                "max_new_tokens": 512,
                "do_sample": True
            }
        )
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # Initialize components
        self.markdown_parser = MarkdownParserWithMetadata()
        self.query_preprocessor = QueryPreprocessor(self.llm)
        self.reranker = ColBERTReranker(device=device)
        
        # Storage
        self.parent_nodes: List[TextNode] = []
        self.child_nodes: List[TextNode] = []
        self.parent_store: Dict[str, TextNode] = {}
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.vector_index: Optional[VectorStoreIndex] = None
    
    def ingest_documents(self, md_filepaths: List[str]):
        """Ingest markdown documents into the pipeline"""
        print(f"Ingesting {len(md_filepaths)} documents...")
        
        all_parent_nodes = []
        all_child_nodes = []
        
        for filepath in md_filepaths:
            print(f"Processing: {filepath}")
            parent_nodes, child_nodes = self.markdown_parser.parse_markdown_file(filepath)
            all_parent_nodes.extend(parent_nodes)
            all_child_nodes.extend(child_nodes)
        
        self.parent_nodes = all_parent_nodes
        self.child_nodes = all_child_nodes
        
        # Build parent store
        self.parent_store = {node.id_: node for node in all_parent_nodes}
        
        print(f"Created {len(all_parent_nodes)} parent chunks and {len(all_child_nodes)} child chunks")
        
        # Build BM25 index on child nodes
        print("Building BM25 index...")
        self.bm25_retriever = BM25Retriever(all_child_nodes)
        
        # Build vector index on child nodes
        print("Building vector index...")
        self.vector_index = VectorStoreIndex(all_child_nodes)
        
        # Persist
        os.makedirs(self.persist_dir, exist_ok=True)
        self.vector_index.storage_context.persist(persist_dir=self.persist_dir)
        
        print("Ingestion complete!")
    
    def _hybrid_retrieve(self, query: str, top_k: int = 50) -> List[Tuple[TextNode, float]]:
        """Perform hybrid retrieval (BM25 + Vector)"""
        # BM25 retrieval
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        
        # Vector retrieval
        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=top_k,
        )
        vector_results = retriever.retrieve(query)
        
        # Combine scores (normalize and average)
        combined_scores = {}
        
        # Add BM25 scores
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            for node, score in bm25_results:
                normalized_score = score / max_bm25 if max_bm25 > 0 else 0
                combined_scores[node.id_] = combined_scores.get(node.id_, 0) + normalized_score * 0.5
        
        # Add vector scores
        if vector_results:
            for node_with_score in vector_results:
                node_id = node_with_score.node.id_
                score = node_with_score.score if hasattr(node_with_score, 'score') else 1.0
                combined_scores[node_id] = combined_scores.get(node_id, 0) + score * 0.5
        
        # Get nodes with combined scores
        node_dict = {node.id_: node for node, _ in bm25_results}
        for node_with_score in vector_results:
            node_dict[node_with_score.node.id_] = node_with_score.node
        
        results = [(node_dict[node_id], score) for node_id, score in combined_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _retrieve_parent_chunks(self, child_nodes: List[TextNode]) -> List[TextNode]:
        """Retrieve parent chunks from child nodes"""
        parent_chunks = []
        seen_parents = set()
        
        for child_node in child_nodes:
            parent_id = child_node.metadata.get('parent_id')
            if parent_id and parent_id not in seen_parents and parent_id in self.parent_store:
                parent_chunks.append(self.parent_store[parent_id])
                seen_parents.add(parent_id)
        
        return parent_chunks
    
    def _generate_structured_answer(self, query: str, context_nodes: List[TextNode]) -> StructuredAnswer:
        """Generate structured answer with citations"""
        # Prepare context
        context_text = "\n\n---\n\n".join([
            f"Source {i+1}:\nDocument: {node.metadata.get('document_name', 'Unknown')}\n"
            f"Section: {node.metadata.get('header_path', 'Unknown')}\n"
            f"Content: {node.get_content()}"
            for i, node in enumerate(context_nodes)
        ])
        
        # Create prompt for structured output
        prompt = f"""Based on the following context, answer the question and provide citations.

Context:
{context_text}

Question: {query}

Provide your answer in the following JSON format:
{{
    "answer": "Your detailed answer here",
    "citations": [
        {{
            "document_name": "filename.md",
            "section_header": "Section title",
            "snippet": "Relevant text snippet from the source"
        }}
    ],
    "confidence": "high/medium/low"
}}

JSON Response:"""
        
        response = self.llm.complete(prompt)
        response_text = str(response)
        
        # Try to parse JSON
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                citations = [Citation(**cite) for cite in result.get('citations', [])]
                return StructuredAnswer(
                    answer=result.get('answer', response_text),
                    citations=citations,
                    confidence=result.get('confidence', 'medium')
                )
        except Exception as e:
            print(f"Failed to parse JSON response: {e}")
        
        # Fallback: create citations from context nodes
        citations = []
        for node in context_nodes[:3]:
            citations.append(Citation(
                document_name=node.metadata.get('document_name', 'Unknown'),
                section_header=node.metadata.get('header_path', 'Unknown'),
                snippet=node.get_content()[:200] + "..."
            ))
        
        return StructuredAnswer(
            answer=response_text,
            citations=citations,
            confidence="medium"
        )
    
    def query(self, question: str, use_hyde: bool = True, 
              use_decomposition: bool = False) -> StructuredAnswer:
        """
        Query the RAG pipeline
        
        Args:
            question: The question to answer
            use_hyde: Whether to use HyDE for query enhancement
            use_decomposition: Whether to decompose complex queries
        
        Returns:
            StructuredAnswer with answer and citations
        """
        print(f"\nProcessing query: {question}")
        
        # Query preprocessing
        queries = [question]
        if use_decomposition:
            print("Decomposing query...")
            queries = self.query_preprocessor.decompose_query(question)
            print(f"Sub-queries: {queries}")
        
        all_retrieved_nodes = []
        
        for query in queries:
            search_query = query
            
            if use_hyde:
                print("Generating hypothetical document...")
                hyde_doc = self.query_preprocessor.generate_hypothetical_document(query)
                search_query = f"{query} {hyde_doc[:200]}"  # Combine query with HyDE
            
            # Hybrid retrieval
            print(f"Performing hybrid retrieval for: {query[:50]}...")
            hybrid_results = self._hybrid_retrieve(search_query, top_k=50)
            
            # Reranking
            print("Reranking results...")
            reranked_results = self.reranker.rerank(query, hybrid_results, top_k=5)
            
            # Get child nodes
            child_nodes = [node for node, _ in reranked_results]
            all_retrieved_nodes.extend(child_nodes)
        
        # Remove duplicates
        unique_nodes = []
        seen_ids = set()
        for node in all_retrieved_nodes:
            if node.id_ not in seen_ids:
                unique_nodes.append(node)
                seen_ids.add(node.id_)
        
        # Retrieve parent chunks (small-to-big)
        print("Retrieving parent chunks (small-to-big)...")
        parent_chunks = self._retrieve_parent_chunks(unique_nodes[:5])
        
        # Generate answer
        print("Generating structured answer...")
        answer = self._generate_structured_answer(question, parent_chunks)
        
        return answer


def main():
    """Main function to demonstrate the pipeline"""
    import pandas as pd
    
    # Configuration
    document_storage_dir = "/home/public/hoangnguyen/TechnicalDocument/Solution/extraction/submission"
    questions_csv = "questions.csv"  # Should be in current directory or provide full path
    
    # Check if document directory exists
    if not os.path.exists(document_storage_dir):
        print(f"Warning: Document directory {document_storage_dir} not found.")
        print("Creating a dummy directory for demonstration...")
        os.makedirs("/tmp/test_docs", exist_ok=True)
        document_storage_dir = "/tmp/test_docs"
        
        # Create a sample markdown file
        with open(os.path.join(document_storage_dir, "sample.md"), "w") as f:
            f.write("""# Sample Document

## Introduction
This is a sample document for testing.

## Section 1
Important information about resistors and their function.

### Subsection 1.1
Resistors limit current flow in circuits.

## Section 2
More information about electronic components.
""")
    
    # Get all markdown files
    md_files = [os.path.join(document_storage_dir, f) 
                for f in os.listdir(document_storage_dir) 
                if f.endswith('.md')]
    
    if not md_files:
        print("No markdown files found!")
        return
    
    print(f"Found {len(md_files)} markdown files")
    
    # Initialize pipeline
    print("\nInitializing Advanced RAG Pipeline...")
    pipeline = AdvancedRAGPipeline(
        llm_model_name="Qwen/Qwen2.5-3B-Instruct",
        embedding_model_name="BAAI/bge-m3",
        device="cuda:0",
        persist_dir="./storage"
    )
    
    # Ingest documents
    print("\nIngesting documents...")
    pipeline.ingest_documents(md_files)
    
    # Load and process questions
    if os.path.exists(questions_csv):
        print(f"\nLoading questions from {questions_csv}...")
        df = pd.read_csv(questions_csv)
        
        results = []
        
        for idx, row in df.iterrows():
            question = row['Question']
            
            print(f"\n{'='*80}")
            print(f"Question {idx + 1}: {question}")
            print('='*80)
            
            # Query the pipeline
            answer = pipeline.query(question, use_hyde=True, use_decomposition=False)
            
            print(f"\nAnswer: {answer.answer}")
            print(f"\nConfidence: {answer.confidence}")
            print(f"\nCitations:")
            for i, citation in enumerate(answer.citations, 1):
                print(f"  {i}. Document: {citation.document_name}")
                print(f"     Section: {citation.section_header}")
                print(f"     Snippet: {citation.snippet[:100]}...")
            
            # For multiple choice questions, try to extract the answer
            # This is a simple heuristic - you might want to improve this
            answer_text = answer.answer.upper()
            choices = []
            for choice in ['A', 'B', 'C', 'D']:
                if choice in row and pd.notna(row[choice]):
                    if str(row[choice]).lower() in answer.answer.lower():
                        choices.append(choice)
            
            if not choices:
                # Try to find letter mentions in the answer
                for choice in ['A', 'B', 'C', 'D']:
                    if f" {choice}" in answer_text or f"({choice})" in answer_text:
                        choices.append(choice)
            
            result = f"{idx + 1},{','.join(choices) if choices else 'A'}"
            results.append(result)
            print(f"\nExtracted answer: {result}")
        
        # Save results
        output_file = "answers.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(results))
        
        print(f"\n{'='*80}")
        print(f"Results saved to {output_file}")
        print('='*80)
    else:
        print(f"\nQuestions file {questions_csv} not found.")
        print("Testing with a sample question...")
        
        test_question = "Linh kiện nào có chức năng hạn chế dòng điện chạy qua mạch?"
        answer = pipeline.query(test_question)
        
        print(f"\nQuestion: {test_question}")
        print(f"Answer: {answer.answer}")
        print(f"\nCitations:")
        for i, citation in enumerate(answer.citations, 1):
            print(f"  {i}. {citation.document_name} - {citation.section_header}")


if __name__ == "__main__":
    main()
