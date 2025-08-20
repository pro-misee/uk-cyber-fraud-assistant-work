# UK Cyber Fraud Assistant Project

## Project Overview

This project develops a specialized conversational AI system to support UK cyber fraud victims through domain-specific fine-tuning of the Mistral-7B language model. The system provides empathetic, accurate, and UK-specific guidance to fraud victims.

## Current Project Status ✅

**Dataset Expansion Completed Successfully**: The training dataset has been expanded to 1000 Q&A pairs through comprehensive multi-source manual curation and AI-powered content generation.

### Latest Dataset Results (V4 - 1000 Pairs)
- **Dataset**: 1000 QA pairs (9x increase from original 111)
- **Multi-source compilation**: Action Fraud, CIFA, Which?, Take Five, NCA, NCSC
- **Gap analysis coverage**: AI-enabled fraud, QR code scams, recovery fraud
- **Quality assurance**: High individual scenario percentage (60%+)
- **Training Ready**: Comprehensive dataset prepared for final model training
- **Dataset Location**: `model_training/master_fraud_qa_dataset_1000_final.json`

## Dataset Information

### Data Sources (V4 - 1000 Pairs)
Authoritative UK fraud guidance from:
1. **Action Fraud** - National fraud reporting centre (265 pairs)
2. **CIFA** - Fraud prevention organisation (106 pairs)
3. **Which** - Consumer protection organisation (73 pairs)
4. **Take Five** - Banking fraud prevention campaign (89 pairs)
5. **National Crime Agency (NCA)** - Filtered consumer content (15 pairs)
6. **National Cyber Security Centre (NCSC)** - Social media safety (10 pairs)
7. **Romance Fraud Content** - Family protection strategies (33 pairs)
8. **Gap Analysis Content** - AI-enabled fraud, QR codes, recovery scams (131 pairs)
9. **Original Dataset** - Foundation fraud guidance (278 pairs)

### Dataset Evolution
- **V1**: 111 QA pairs → Overfitting after epoch 3
- **V2**: 278 QA pairs → Initial overfitting issues
- **V3**: 278 QA pairs with optimized parameters → ✅ Successful training
- **V4**: 1000 QA pairs → ✅ Comprehensive dataset ready for production training

### Dataset Files
- `model_training/master_fraud_qa_dataset_1000_final.json` - **Current training dataset (1000 pairs)**
- `model_training/master_fraud_qa_dataset.json` - Previous dataset (278 pairs)
- `manual_scraped_content/` - Manual curation and processing of authoritative sources
- `merged/` - Contains consolidated and processed datasets
- `data_sources_v1/`, `data_sources_v2/`, `data_sources_v3/` - Raw scraped data

## Technical Architecture

### Base Model
**Mistral-7B-Instruct-v0.3**
- Selected for superior instruction-following capabilities
- Enhanced reasoning for nuanced fraud scenarios
- Minimal content filtering suitable for fraud guidance
- 7B parameters provide optimal balance of performance and efficiency

### Fine-Tuning Configuration
**Unsloth-Optimized Training**:
- **LoRA Configuration**: r=48, alpha=96 (optimized through iterations)
- **Learning Rate**: 2e-5 (conservative to prevent overfitting)
- **Training**: Full precision (bfloat16) with 5 epochs
- **Early Stopping**: Patience=3 (though not triggered in final run)
- **Batch Size**: Effective batch size of 16 (2×8 gradient accumulation)

### Training Evolution
1. **V1**: r=64, lr=1e-4 → Overfitting at epoch 3
2. **V2**: r=64, lr=1e-4 → Early stopping at epoch 4  
3. **V3**: r=48, lr=2e-5 → ✅ Successful training, no overfitting

## File Structure

```
uk-cyber-fraud-assistant/
├── CLAUDE.md                          # This file - project overview
├── PROJECT_LOG.md                     # Comprehensive development log (all changes/errors/results)
├── METHODOLOGY.md                     # Academic methodology documentation
├── README.md                          # Project introduction
├── Unsloth_Fine_Tuning.ipynb          # Original training notebook (111 samples)
├── Unsloth_Fine_Tuning_v2.ipynb      # Updated training notebook (278 samples)
├── requirements.txt                   # Python dependencies
│
├── model_training/                    # Training datasets
│   ├── master_fraud_qa_dataset_1000_final.json  # Current dataset (1000 pairs)
│   ├── master_fraud_qa_dataset.json  # Previous dataset (278 pairs)
│   └── [other training files]
│
├── manual_scraped_content/            # V4 dataset expansion
│   ├── actionfraud/processed/         # Action Fraud Q&A pairs (265)
│   ├── cifa/processed/                # CIFA financial crime pairs (106)
│   ├── which/processed/               # Which? consumer protection (73)
│   ├── takefive/processed/            # Take Five banking fraud (89)
│   ├── nca/processed/                 # NCA filtered content (15)
│   ├── ncsc/processed/                # NCSC social media safety (10)
│   ├── extra/processed/               # Romance fraud content (33)
│   ├── claude_generated/              # Gap analysis pairs (56)
│   └── final_completion_75_pairs.json # Final completion to 1000
│
├── data_sources_v[1-3]/              # Raw scraped data by version
│   ├── actionfraud/
│   ├── citizensadvice/
│   ├── fca/
│   ├── getsafeonline/
│   ├── ncsc/
│   ├── policeuk/
│   ├── ukfinance/
│   ├── victimsupport/
│   └── which/
│
├── merged/                           # Processed and consolidated datasets
├── scripts/                         # Data processing scripts
└── notebooks/                       # Demo and development notebooks
```

## Key Commands and Workflows

### For Training
```bash
# Install dependencies
pip install -r requirements.txt

# Training is done in Google Colab with GPU:
# 1. Upload Unsloth_Fine_Tuning_v2.ipynb to Colab
# 2. Mount Google Drive
# 3. Upload master_fraud_qa_dataset.json to Drive
# 4. Run all cells for training
```

### For Local Deployment
The trained model supports multiple deployment options:

**LM Studio (GUI)**:
- Load the GGUF model file
- Use system prompt for UK fraud assistance context
- Temperature: 0.1, Top-p: 0.9, Max tokens: 350

**Ollama (CLI)**:
```bash
# Create model from GGUF and Modelfile
ollama create uk-fraud-assistant -f Modelfile
ollama run uk-fraud-assistant
```

### Model Performance
The final model demonstrates:
- **UK-specific accuracy**: Correct Action Fraud contact details (0300 123 2040)
- **Empathetic responses**: Appropriate emotional support for victims
- **Practical guidance**: Clear actionable steps for fraud reporting
- **EOS token completion**: Proper response termination in LM Studio

## Development Tracking and Logging

This project maintains comprehensive logging of all development activities in `PROJECT_LOG.md`:

### Tracked Information
- **Training Attempts**: All V1, V2, V3 iterations with complete metrics
- **Parameter Changes**: LoRA configurations, learning rates, batch sizes
- **Dataset Evolution**: From 111 → 278 samples with quality improvements
- **Error Analysis**: Overfitting issues, parameter optimization, solutions
- **Results Documentation**: Loss trajectories, convergence patterns, performance
- **Technical Decisions**: Model selection rationale, deployment choices
- **Deployment Issues**: Export challenges, local testing results

### Logging Protocol for Claude Code
**⚠️ IMPORTANT: Claude Code must update `PROJECT_LOG.md` for every significant change:**

- **All training attempts** with parameters, results, and analysis
- **Dataset modifications** including size, quality, and processing changes
- **Parameter adjustments** with rationale and impact assessment
- **Errors encountered** with detailed analysis and resolution steps
- **Model performance** metrics, validation results, and deployment testing
- **Technical decisions** including architecture choices and deployment strategies
- **Deployment issues** with debugging steps and resolution outcomes

**Each entry must include**: Timestamp, context, specific changes, results, and lessons learned.

### Current Status Summary
- **✅ Dataset Expansion Complete**: V4 successful with 1000 high-quality Q&A pairs
- **✅ Multi-source Integration**: Comprehensive coverage from 8 authoritative UK sources
- **✅ Gap Analysis Complete**: AI-enabled fraud, QR codes, recovery scams coverage
- **✅ Documentation Updated**: METHODOLOGY.md and project files reflect current state
- **🔄 Ready for Training**: 1000-pair dataset prepared for final Mistral-7B fine-tuning

## Current Challenges and Solutions

### ✅ Solved: Overfitting Prevention
- **Problem**: Previous versions overfitted after epoch 2-3
- **Solution**: Reduced learning rate (2e-5), adjusted LoRA rank (48), increased patience
- **Result**: Stable training with continuous improvement across all 5 epochs
- **Logged**: Complete parameter evolution in PROJECT_LOG.md

### ✅ Solved: Dataset Scale (Comprehensive)
- **Problem**: Initial 111 samples insufficient for 7B model
- **Solution**: Comprehensive expansion to 1000 samples through multi-source manual curation
- **Result**: Optimal dataset size for robust model training and generalization
- **Sources**: Action Fraud, CIFA, Which?, Take Five, NCA, NCSC, gap analysis content
- **Logged**: Complete dataset expansion methodology and quality assessment

### ✅ Solved: Content Quality and Coverage
- **Problem**: Limited fraud type coverage and individual scenarios
- **Solution**: Manual curation with 60%+ individual scenario focus, emerging fraud coverage
- **Result**: Comprehensive fraud guidance including AI-enabled scams, QR codes, recovery fraud
- **Quality**: UK-specific guidance with correct reporting procedures (Action Fraud 0300 123 2040)
- **Logged**: Quality assessment and gap analysis documented

### 🔄 Current: Production Training
- **Status**: 1000-pair dataset compiled and ready for final training
- **Next**: Execute training with optimized parameters on expanded dataset
- **Target**: Production-ready model with comprehensive fraud coverage
- **Deployment**: GGUF export for LM Studio and Ollama local inference

## Next Steps

1. **Execute Final Training**: Train Mistral-7B on 1000-pair dataset with optimized parameters
2. **Model Validation**: Comprehensive testing across all fraud categories and scenarios
3. **GGUF Export**: Package trained model for local deployment (LM Studio, Ollama)
4. **Performance Evaluation**: Systematic assessment of UK-specific fraud guidance accuracy
5. **Production Deployment**: Create deployment guides and user documentation
6. **Future Scaling**: Framework established for 2000+ pairs enterprise-level expansion

## Research Impact

This project contributes to:
- **AI for Social Good**: Specialized AI systems for vulnerable populations
- **Domain-Specific Fine-Tuning**: Methodology for crisis support applications  
- **Fraud Prevention**: Accessible AI-powered victim support tools
- **Academic Research**: Open methodology for defensive security applications

## Contact and Usage

This is a defensive security project designed to help fraud victims. The methodology and code are available for:
- Academic research and education
- Fraud prevention organizations
- Crisis support system development
- General cybersecurity awareness

**Model Capabilities**: UK-specific fraud guidance, victim support, reporting procedures
**Limitations**: Academic project, not replacement for professional legal advice
**Ethics**: Designed for defensive purposes only, focused on victim support

**Remember**: Always update the PROJECT_LOG.md and CLAUDE.md whenever any modification is made.
---

*Last Updated: August 2025*
*Training Status: Complete and Successful*
*Model Version: V3 (278 samples, stable convergence)*