#!/usr/bin/env python3
"""
Fix JSON format for training - replace "" values with empty strings
"""

import json
import pandas as pd

def fix_json_format():
    """Fix the JSON format for training compatibility"""
    
    # Load the deduplicated CSV
    df = pd.read_csv('merged/deduplicated_fraud_qa_dataset.csv')
    
    print(f"📂 Loaded {len(df)} samples")
    
    # Fill "" values in 'input' column with empty strings
    df['input'] = df['input'].fillna('')
    
    # Convert to records (list of dicts)
    training_data = df.to_dict('records')
    
    # Save the fixed JSON
    with open('merged/training_ready_fraud_qa_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Fixed JSON saved: merged/training_ready_fraud_qa_dataset.json")
    print(f"📊 Total samples: {len(training_data)}")
    
    # Verify the first sample
    sample = training_data[0]
    print(f"\n🔍 Sample structure:")
    print(f"Keys: {list(sample.keys())}")
    print(f"Input field: '{sample['input']}'")
    print(f"Instruction: {sample['instruction'][:80]}...")
    
    return len(training_data)

if __name__ == "__main__":
    print("🔧 Fixing JSON format for training...")
    count = fix_json_format()
    print(f"🎯 Ready for training with {count} samples!")