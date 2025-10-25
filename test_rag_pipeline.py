"""
Test script for Advanced RAG Pipeline

This script tests the pipeline with sample documents to ensure everything works.
"""

import os
import sys
from pathlib import Path

def create_sample_documents():
    """Create sample markdown documents for testing"""
    test_dir = "/tmp/test_documents"
    os.makedirs(test_dir, exist_ok=True)
    
    # Sample document 1: Electronics basics
    doc1 = """# Electronics Basics

## Chapter 1: Basic Components

### Resistors
Resistors are passive electronic components that limit the flow of electric current in a circuit. They have a resistance value measured in ohms (Ω).

#### Function of Resistors
The main function of a resistor is to restrict or limit the current flowing through a circuit. This is achieved through the resistance it provides to the flow of electrons.

#### Types of Resistors
- Fixed resistors: Have a constant resistance value
- Variable resistors: Resistance can be adjusted
- Special resistors: Light-dependent, temperature-dependent, etc.

### Capacitors
Capacitors store electrical energy in an electric field. They consist of two conductive plates separated by an insulating material called a dielectric.

#### Function of Capacitors
Capacitors store and release electrical energy. They can block DC current while allowing AC current to pass.

### Transistors
Transistors are semiconductor devices used to amplify or switch electronic signals. They are the fundamental building blocks of modern electronics.

#### Function of Transistors
Transistors can act as amplifiers or switches in electronic circuits. They control the flow of current based on input signals.

### Diodes
Diodes are semiconductor devices that allow current to flow in only one direction. They are used for rectification, signal demodulation, and voltage regulation.

#### Function of Diodes
Diodes allow current to flow in one direction (forward bias) while blocking it in the reverse direction (reverse bias).

## Chapter 2: Circuit Analysis

### Ohm's Law
Ohm's Law states that the current through a conductor between two points is directly proportional to the voltage across the two points.

Formula: V = I × R
Where:
- V is voltage in volts (V)
- I is current in amperes (A)
- R is resistance in ohms (Ω)

### Power Calculations
The power consumed by an electrical component can be calculated using several formulas:
- P = V × I (Power = Voltage × Current)
- P = I² × R (Power = Current squared × Resistance)
- P = V² / R (Power = Voltage squared / Resistance)

### Example: Power Consumption
If a circuit has a voltage source of 230 VAC and the maximum current is 0.25 A:
P = V × I = 230 V × 0.25 A = 57.5 W

This means the maximum power consumption would be approximately 57.5 watts.
"""
    
    # Sample document 2: PowerPoint/Presentation software
    doc2 = """# Presentation Software Guide

## Introduction
This guide covers the features and capabilities of modern presentation software like Microsoft PowerPoint.

## Chapter 1: Creating Presentations

### Inserting Content into Slides
Modern presentation software allows you to insert various types of content:

#### Text Content
You can insert and format text in multiple ways:
- Title text boxes
- Body text boxes
- Text from external files

#### Images and Graphics
Images can be inserted from:
- Local files (JPG, PNG, GIF, etc.)
- Online sources
- Screenshots
- Clip art

#### Multimedia Content
You can insert rich multimedia:
- Audio files (MP3, WAV, etc.)
- Video files (MP4, AVI, etc.)
- Embedded YouTube videos
- Audio recordings

#### Other Elements
Additional elements that can be inserted:
- Tables and charts
- SmartArt graphics
- Shapes and icons
- Equations and symbols

## Chapter 2: Customizing Presentations

### Slide Design
You can customize the appearance of your slides:

#### Background Customization
- Change slide background colors
- Apply gradient backgrounds
- Use images as backgrounds
- Apply textures

#### Templates
- Choose from built-in templates
- Download custom templates
- Create your own templates
- Apply themes consistently

### Animation and Effects

#### Text Effects
Create engaging text animations:
- Entrance effects (fade in, fly in, etc.)
- Emphasis effects (bold, color change, etc.)
- Exit effects (fade out, fly out, etc.)

#### Object Animation
Animate images, shapes, and other objects:
- Motion paths
- Rotation effects
- Scaling effects
- Custom animation timings

#### Slide Transitions
Add transitions between slides:
- Fade transitions
- Push transitions
- Wipe effects
- 3D transitions

### Summary of Features
In summary, modern presentation software allows you to:
1. Insert text, images, audio, and video content
2. Change background colors and apply templates
3. Create effects for text and images
4. Add transitions between slides

This comprehensive feature set makes it possible to create professional and engaging presentations.
"""
    
    # Sample document 3: More electronics
    doc3 = """# Advanced Electronics

## Power Supplies

### AC/DC Power Supplies
Power supplies convert AC mains voltage to DC voltage for electronic devices.

#### Specifications
Common specifications for power supplies:
- Input voltage: 230 VAC or 110 VAC
- Output voltage: 5V, 12V, 24V, etc.
- Maximum current rating: 0.25A to 10A or more
- Power rating: Calculated as V × I

### Current Limiting
Electronic devices often include current limiting mechanisms to prevent damage:
- Resistors for passive current limiting
- Active current limiting circuits
- Fuses for overcurrent protection

The primary component for limiting current in a simple circuit is the resistor.
"""
    
    # Write documents
    with open(os.path.join(test_dir, "electronics_basics.md"), "w", encoding="utf-8") as f:
        f.write(doc1)
    
    with open(os.path.join(test_dir, "presentation_software.md"), "w", encoding="utf-8") as f:
        f.write(doc2)
    
    with open(os.path.join(test_dir, "advanced_electronics.md"), "w", encoding="utf-8") as f:
        f.write(doc3)
    
    print(f"Created sample documents in {test_dir}")
    return test_dir


def test_pipeline():
    """Test the RAG pipeline"""
    try:
        from advanced_rag_pipeline import AdvancedRAGPipeline
        import pandas as pd
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("\nPlease install required packages:")
        print("pip install -r requirements_rag.txt")
        return False
    
    # Create sample documents
    doc_dir = create_sample_documents()
    
    # Get markdown files
    md_files = [os.path.join(doc_dir, f) for f in os.listdir(doc_dir) if f.endswith('.md')]
    print(f"\nFound {len(md_files)} markdown files:")
    for f in md_files:
        print(f"  - {f}")
    
    # Initialize pipeline
    print("\n" + "="*80)
    print("Initializing Advanced RAG Pipeline...")
    print("="*80)
    
    try:
        pipeline = AdvancedRAGPipeline(
            llm_model_name="Qwen/Qwen2.5-3B-Instruct",
            embedding_model_name="BAAI/bge-m3",
            device="cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
            persist_dir="./test_storage"
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("\nThis might be due to:")
        print("1. Missing GPU (trying to use CUDA)")
        print("2. Model download issues")
        print("3. Insufficient memory")
        return False
    
    # Ingest documents
    print("\n" + "="*80)
    print("Ingesting Documents...")
    print("="*80)
    
    try:
        pipeline.ingest_documents(md_files)
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return False
    
    # Test questions
    print("\n" + "="*80)
    print("Testing with Sample Questions...")
    print("="*80)
    
    test_questions = [
        "Linh kiện nào có chức năng hạn chế dòng điện chạy qua mạch?",
        "Tóm tắt các thành phần có thể chèn vào Slide và cách tùy chỉnh giao diện/hiệu ứng trình chiếu.",
        "Nếu dòng điện cực đại của nguồn 230 VAC là 0.25 A, công suất tiêu thụ cực đại của RCE khoảng bao nhiêu?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print('='*80)
        
        try:
            answer = pipeline.query(question, use_hyde=False, use_decomposition=False)
            
            print(f"\nAnswer: {answer.answer}")
            print(f"\nConfidence: {answer.confidence}")
            print(f"\nCitations ({len(answer.citations)}):")
            for j, citation in enumerate(answer.citations, 1):
                print(f"  {j}. Document: {citation.document_name}")
                print(f"     Section: {citation.section_header}")
                print(f"     Snippet: {citation.snippet[:150]}...")
        except Exception as e:
            print(f"Error processing question: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)
    
    return True


if __name__ == "__main__":
    print("Advanced RAG Pipeline - Test Script")
    print("="*80)
    
    success = test_pipeline()
    
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Tests failed. Please check the errors above.")
        sys.exit(1)
