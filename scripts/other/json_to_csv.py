#!/usr/bin/env python3
"""
Convert consolidated_fraud_qa_dataset.json to CSV format for easy analysis and deduplication
"""

import json
import pandas as pd
from pathlib import Path

def convert_json_to_csv():
    """Convert JSON dataset to CSV with proper formatting"""
    
    # Input and output paths
    json_file = Path("merged/consolidated_fraud_qa_dataset.json")
    csv_file = Path("merged/consolidated_fraud_qa_dataset.csv")
    
    if not json_file.exists():
        print(f"❌ Error: {json_file} not found!")
        return False
    
    try:
        # Load JSON data
        print(f"📂 Loading data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ Error: JSON file should contain a list of Q&A pairs")
            return False
        
        print(f"✅ Loaded {len(data)} Q&A pairs")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Display column information
        print(f"\n📊 Dataset Structure:")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        
        # Check for missing values
        print(f"\n🔍 Missing Values Check:")
        missing_counts = df.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  - {col}: {count} missing values")
        
        if missing_counts.sum() == 0:
            print("  ✅ No missing values found")
        
        # Display sample data
        print(f"\n📋 Sample Data Preview:")
        print("First Q&A pair:")
        print(f"Instruction: {df.iloc[0]['instruction'][:100]}...")
        print(f"Output: {df.iloc[0]['output'][:100]}...")
        
        # Save to CSV
        print(f"\n💾 Saving to CSV: {csv_file}")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"✅ Successfully converted to CSV!")
        print(f"📄 Output file: {csv_file}")
        print(f"📈 Total rows: {len(df)}")
        
        # Create additional analysis
        print(f"\n📊 Quick Analysis:")
        
        # Source distribution
        if 'source_document' in df.columns:
            source_counts = df['source_document'].value_counts()
            print(f"Top 5 source documents:")
            for source, count in source_counts.head().items():
                print(f"  - {source}: {count}")
        
        # Response length analysis
        df['instruction_length'] = df['instruction'].str.len()
        df['output_length'] = df['output'].str.len()
        
        print(f"\nText Length Statistics:")
        print(f"Average instruction length: {df['instruction_length'].mean():.1f} chars")
        print(f"Average output length: {df['output_length'].mean():.1f} chars")
        print(f"Max instruction length: {df['instruction_length'].max()} chars")
        print(f"Max output length: {df['output_length'].max()} chars")
        
        # Save analysis report
        analysis_report = {
            "total_samples": len(df),
            "columns": list(df.columns),
            "missing_values": missing_counts.to_dict(),
            "text_length_stats": {
                "avg_instruction_length": df['instruction_length'].mean(),
                "avg_output_length": df['output_length'].mean(),
                "max_instruction_length": df['instruction_length'].max(),
                "max_output_length": df['output_length'].max()
            }
        }
        
        if 'source_document' in df.columns:
            analysis_report["source_distribution"] = source_counts.to_dict()
        
        report_file = Path("merged/dataset_analysis_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Analysis report saved to: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting JSON to CSV conversion...")
    print("=" * 50)
    
    success = convert_json_to_csv()
    
    print("=" * 50)
    if success:
        print("✅ Conversion completed successfully!")
        print("\nFiles created:")
        print("  📄 merged/consolidated_fraud_qa_dataset.csv")
        print("  📋 merged/dataset_analysis_report.json")
        print("\nYou can now:")
        print("  1. Open the CSV in Excel/Google Sheets for manual review")
        print("  2. Use pandas for deduplication and analysis")
        print("  3. Sort and filter data easily")
    else:
        print("❌ Conversion failed!")