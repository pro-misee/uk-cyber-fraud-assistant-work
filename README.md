# UK Cyber Fraud Assistant - Fine-Tuned AI Support System

## Project Overview (Last training)

A specialized conversational AI system designed to support UK cyber fraud victims through domain-specific fine-tuning of Mistral-7B. The project combines comprehensive data collection, AI-powered Q&A generation, and advanced fine-tuning techniques to create an empathetic, accurate fraud victim support assistant.

## Current Status: Training Complete ✅

**Successfully trained Mistral-7B model** with excellent performance and no overfitting.

### Latest Results (V3 Training)
- **Dataset Size**: 278 high-quality Q&A pairs (2.5x increase from original)
- **Training Status**: Complete and successful - no overfitting detected
- **Final Validation Loss**: 1.147 (continuous improvement across all 5 epochs)
- **Model Performance**: Stable convergence with excellent generalization

## Data Sources & Processing

### Authoritative UK Sources (278 Total Q&A Pairs)
- **Action Fraud**: UK's national fraud reporting centre
- **GetSafeOnline**: UK's leading internet safety resource  
- **FCA (Financial Conduct Authority)**: Financial scams and consumer protection
- **UK Finance**: Industry fraud prevention strategies
- **Which**: Consumer protection guidance and fraud prevention
- **Citizens Advice**: Consumer guidance and support
- **Additional Sources**: NCSC, Police.uk, Victim Support

### Enhanced Dataset Characteristics

### Quality Characteristics
- **Victim-focused questions**: All questions written from fraud victim perspective
- **Empathetic responses**: Supportive, non-judgmental tone throughout
- **UK-specific guidance**: Includes proper UK contact numbers and procedures
- **Source attribution**: Every Q&A pair linked to authoritative source material
- **Training-ready format**: Structured in Alpaca format for LLaMA 2 fine-tuning

## Technical Implementation

### 1. Web Scraping Framework (`scripts/`)

Built a scraping framework that works across different UK fraud guidance websites:

**Key Features:**
- **Configurable source management**: Easy addition of new fraud guidance websites
- **Intelligent content extraction**: Automatically extracts meaningful content from various site structures
- **Smart link following**: Discovers and scrapes relevant linked pages using fraud-related keywords
- **Respectful scraping**: 3-second delays between requests, respects robots.txt
- **Structured output**: Individual JSON files per page plus compiled datasets

**Site Configurations:**
```python
SITE_CONFIGS = {
    'actionfraud': {...},    # UK's national fraud reporting centre
    'getsafeonline': {...},  # UK's leading internet safety resource
    'fca': {...},           # Financial Conduct Authority 
    'ukfinance': {...},     # UK Finance industry body
    'which': {...},         # Consumer protection guidance
    'citizensadvice': {...} # Citizens Advice consumer guidance
}
```

### 2. Q&A Generation Process

Used Gemini AI with carefully crafted prompts to generate victim-focused Q&A pairs:

**Generation Pipeline:**
```
Compiled dataset → Gemini AI → Victim-focused Q&A pairs → JSON validation → Training dataset
```

**Process Details:**
- **Chunking strategy**: Split large datasets into 2-3 documents per chunk for optimal quality
- **Victim perspective**: Generated questions from fraud victim's point of view
- **Empathetic responses**: 100-300 word supportive answers grounded in source material
- **UK-specific guidance**: Embedded proper contact numbers and UK procedures
- **Quality validation**: Ensured proper JSON format for downstream processing

**Output Format:**
```json
[
  {
    "instruction": "I think I've been scammed online — what should I do immediately?",
    "input": "",
    "output": "I understand how distressing this must be for you. If you believe you've been scammed, here are the immediate steps you should take...",
    "source_document": "Action Fraud - What to do if you've been scammed",
    "source_url": "https://www.actionfraud.police.uk/...",
    "chunk_number": 1,
    "document_index": 1,
    "generated_by": "gemini"
  }
]
```

### 3. Advanced Model Training (Mistral-7B)

Successfully fine-tuned Mistral-7B-Instruct-v0.3 with optimized parameters:

**Training Configuration:**
- **Base Model**: Mistral-7B-Instruct-v0.3 (state-of-the-art instruction following)
- **Method**: Full-precision LoRA with Unsloth optimization (2x speed improvement)
- **Dataset**: 278 Q&A pairs from UK fraud guidance sources
- **Training Results**: Stable convergence, no overfitting across 5 epochs
- **Optimization**: r=48, alpha=96, learning_rate=2e-5
- **Performance**: Validation loss improved continuously (1.509 → 1.147)

## Project Structure

```
uk-fraud-chatbot/
├── data_sources/                    # Organised by source website
│   ├── actionfraud/
│   │   ├── scraped/                # Raw scraped content
│   │   └── processed/gemini_qa_pairs/  # Generated Q&A pairs
│   ├── getsafeonline/
│   ├── fca/
│   ├── ukfinance/
│   ├── which/
│   └── citizensadvice/
├── scripts/
│   ├── scraper.py                  # Main scraping orchestrator
│   ├── site_scrapers.py            # Generic scraping framework + configurations
│   ├── combine_datasets.py         # Dataset combination utilities
├── model_training/
│   ├── master_fraud_qa_dataset.json    # Combined training dataset (278 Q&A pairs)
│   ├── train_fraud_qa_dataset.json     # Training split (222 samples)
│   ├── val_fraud_qa_dataset.json       # Validation split (56 samples)
│   └── dataset_info.json              # Dataset metadata
├── notebooks/
│   ├── Demo_Chatbot_Colab.ipynb              # Interactive demo notebook
├── Unsloth_Fine_Tuning.ipynb          # Original training notebook (111 samples)
├── Unsloth_Fine_Tuning_v2.ipynb       # Current training notebook (278 samples)
├── CLAUDE.md                           # Comprehensive project documentation
├── QA_prompt_engineering.txt      # Gemini Q&A generation instructions
├── requirements.txt                # Python dependencies
└── README.md                      # This documentation
```

## Usage Instructions

### 1. Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv fraud_chatbot_env
source fraud_chatbot_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Collection
```bash
cd scripts

# Scrape specific source (modify scraper.py line 81 to change source)
python scraper.py  # Currently configured for UK Finance

# Available sources: actionfraud, getsafeonline, fca, ukfinance, which, citizensadvice
```

### 3. Q&A Generation
```bash
# Use the Gemini prompt with your scraped data
# 1. Copy content from model_processing_prompt.txt
# 2. Input scraped dataset into Gemini AI
# 3. Save generated Q&A pairs to data_sources/[source]/processed/gemini_qa_pairs/
```

### 4. Model Training
```bash
# Use the Unsloth-optimized training notebook in Google Colab
# Unsloth_Fine_Tuning_v2.ipynb contains the complete Mistral-7B training pipeline
# Requires GPU (recommended: A100 or L4) for training
```

### 5. Local Deployment
```bash
# Deploy using LM Studio (GUI)
# 1. Download GGUF model file
# 2. Load in LM Studio with system prompt for fraud assistance
# 3. Configure: Temperature 0.1, Top-p 0.9, Max tokens 350

# Deploy using Ollama (CLI)
ollama create uk-fraud-assistant -f Modelfile
ollama run uk-fraud-assistant
```

## Key Design Decisions

### 1. Generic Scraping Framework
I built a reusable scraper that works across different fraud guidance sites because fraud information comes from multiple authoritative sources. This framework allows easy expansion to new sources without rewriting scraping logic.

### 2. Mistral-7B Selection
Selected Mistral-7B-Instruct-v0.3 over LLaMA alternatives for superior instruction-following capabilities, enhanced reasoning for nuanced fraud scenarios, and minimal content filtering suitable for fraud guidance training.

### 3. Full-Precision LoRA Training
Chose full-precision LoRA over QLoRA to maximize model quality, leveraging available GPU resources for optimal performance without quantization-induced degradation.

### 4. Source-Specific Organisation
Organized data by source website rather than topic to maintain proper attribution and allow for source-specific processing while enabling easy combination for training.

### 5. UK-Specific Focus
Ensured all responses include proper UK contact numbers, procedures, and legal context because generic responses are unhelpful in crisis situations. Fraud victims need specific, actionable guidance.

## Results and Achievements

### Enhanced Dataset Quality
- **278 total Q&A pairs** across 8+ authoritative UK sources (2.5x increase)
- **100% victim-focused questions** written from fraud victim perspective  
- **UK-specific guidance** with proper contact numbers and procedures
- **Source attribution** linking every Q&A pair to authoritative material
- **Training-ready format** optimized for Mistral-7B fine-tuning

### Superior Model Performance  
- **Successfully fine-tuned Mistral-7B** with stable convergence and no overfitting
- **Excellent training metrics**: Validation loss improved continuously (1.509 → 1.147)
- **UK-specific knowledge**: Accurate Action Fraud contact details (0300 123 2040)
- **Empathetic responses**: Appropriate emotional support for fraud victims
- **Proper completion**: EOS token generation indicating healthy model behavior

### Advanced Technical Infrastructure
- **Unsloth-optimized training**: 2x speed improvement with 70% memory reduction
- **Full-precision LoRA**: Maximum quality without quantization degradation
- **Multi-platform deployment**: Support for LM Studio, Ollama, and GGUF format
- **Scalable data pipeline**: Enhanced Q&A generation with 5-8x extraction efficiency

## Data Sources & Attribution

### Authoritative UK Fraud Guidance Sources
- **Action Fraud**: UK's national fraud and cybercrime reporting centre (actionfraud.police.uk)
- **GetSafeOnline**: UK's leading internet safety website (getsafeonline.org)
- **FCA**: Financial Conduct Authority - UK financial regulator (fca.org.uk)
- **UK Finance**: Industry body representing UK financial services (ukfinance.org.uk)
- **Which**: UK consumer protection organization (which.co.uk)
- **Citizens Advice**: UK consumer guidance organization (citizensadvice.org.uk)

## Future Enhancements

### Immediate Priorities
- **Scale dataset to 1000+ samples**: Target production-ready model with comprehensive fraud coverage
- **Enhanced evaluation framework**: Systematic testing across diverse fraud scenarios
- **API development**: RESTful interface for integration with existing support systems

### Long-term Goals
- **Multi-modal capabilities**: Image and document analysis for fraud detection
- **Real-time updates**: Dynamic knowledge integration for emerging fraud patterns
- **Mobile deployment**: Accessible crisis support applications
- **Professional integration**: Tools for fraud prevention and law enforcement

## Project Development Logging

This project maintains comprehensive development logs tracking all changes, decisions, errors, and results:

### PROJECT_LOG.md - Complete Development History
- **Training Iterations**: V1 → V2 → V3 with detailed metrics and parameter changes
- **Error Analysis**: Overfitting issues, parameter optimization strategies, solutions
- **Dataset Evolution**: From 111 → 278 samples with enhancement methods
- **Technical Decisions**: Model selection rationale, deployment choices, lessons learned
- **Performance Tracking**: All loss trajectories, convergence patterns, validation results
- **Deployment Issues**: Export challenges, testing results, debugging steps

### Key Logged Events
- **V1 Training**: 111 samples, overfitting at epoch 3 (val loss: 1.058 → 1.211)
- **V2 Training**: 278 samples, early stopping at epoch 4, still overfitting
- **V3 Success**: Optimized parameters (r=48, lr=2e-5), stable convergence (val loss: 1.147)
- **Model Deployment**: LM Studio success, EOS token validation, response quality testing

## Model Access and Documentation

- **Development Log**: `PROJECT_LOG.md` (comprehensive change tracking with timestamps)
- **Training Notebooks**: `Unsloth_Fine_Tuning_v2.ipynb` (current), `Unsloth_Fine_Tuning.ipynb` (original)
- **Project Documentation**: `CLAUDE.md` (comprehensive overview), `METHODOLOGY.md` (academic methodology)
- **Dataset**: `model_training/master_fraud_qa_dataset.json` (278 Q&A pairs)
- **Deployment**: GGUF format for local inference, Ollama and LM Studio compatible

---