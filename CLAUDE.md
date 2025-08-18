# UK Cyber Fraud Assistant Project

## Project Overview

This project develops a specialized conversational AI system to support UK cyber fraud victims through domain-specific fine-tuning of the Mistral-7B language model. The system provides empathetic, accurate, and UK-specific guidance to fraud victims.

## Current Project Status ✅

**Training Completed Successfully**: The model has been fine-tuned with excellent results showing no overfitting and stable convergence.

### Latest Training Results (V3)
- **Dataset**: 278 QA pairs (2.5x increase from original 111)
- **Training Loss**: 2.187 → 0.644 (smooth progression)
- **Validation Loss**: 1.509 → 1.147 (continuous improvement)
- **No Overfitting**: Validation loss improved throughout all 5 epochs
- **Training Time**: 464.8 seconds
- **Final Model**: Stable, well-generalized fraud support assistant

## Dataset Information

### Data Sources
Authoritative UK fraud guidance from:
1. **Action Fraud** - National fraud reporting centre  
2. **GetSafeOnline** - Internet safety resource
3. **Financial Conduct Authority (FCA)** - Financial services regulator
4. **UK Finance** - Financial services industry body
5. **Which** - Consumer protection organisation
6. **Citizens Advice** - Consumer guidance organisation

### Dataset Evolution
- **V1**: 111 QA pairs → Overfitting after epoch 3
- **V2**: 278 QA pairs → Initial overfitting issues
- **V3**: 278 QA pairs with optimized parameters → ✅ Success

### Dataset Files
- `model_training/master_fraud_qa_dataset.json` - Current training dataset (278 pairs)
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
│   ├── master_fraud_qa_dataset.json  # Current dataset (278 pairs)
│   └── [other training files]
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
- **✅ Training Complete**: V3 successful with no overfitting (validation loss: 1.147)
- **✅ Model Deployed**: Successfully tested in LM Studio with proper EOS completion
- **✅ Documentation**: Comprehensive logs and methodology documentation
- **🔄 Export Finalization**: GGUF packaging for broader deployment

## Current Challenges and Solutions

### ✅ Solved: Overfitting Prevention
- **Problem**: Previous versions overfitted after epoch 2-3
- **Solution**: Reduced learning rate (2e-5), adjusted LoRA rank (48), increased patience
- **Result**: Stable training with continuous improvement across all 5 epochs
- **Logged**: Complete parameter evolution in PROJECT_LOG.md

### ✅ Solved: Dataset Scale
- **Problem**: Initial 111 samples insufficient for 7B model
- **Solution**: Expanded to 278 samples using enhanced QA generation
- **Result**: Better model performance and generalization
- **Logged**: Dataset expansion process and impact analysis

### 🔄 Current: Model Export and Deployment
- **Issue**: Zip file creation for model download
- **Status**: Model files saved successfully, working on export process
- **Next**: Finalize GGUF export and local deployment package
- **Logged**: Export challenges and debugging steps documented

## Next Steps

1. **Complete Model Export**: Finalize GGUF model packaging for local deployment
2. **Deployment Testing**: Validate model performance in LM Studio and Ollama
3. **Dataset Expansion**: Scale to 1000+ QA pairs for production-ready model
4. **Evaluation Framework**: Systematic testing across fraud scenarios
5. **Documentation**: Complete user guides for model deployment and usage

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