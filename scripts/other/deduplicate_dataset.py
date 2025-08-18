#!/usr/bin/env python3
"""
Remove duplicated rows from the consolidated fraud Q&A dataset
"""

import pandas as pd
import json
from pathlib import Path

def deduplicate_dataset():
    """Remove duplicates from the dataset with detailed reporting"""
    
    # Input and output paths
    input_csv = Path("merged/consolidated_fraud_qa_dataset.csv")
    output_csv = Path("merged/deduplicated_fraud_qa_dataset.csv")
    output_json = Path("merged/deduplicated_fraud_qa_dataset.json")
    report_file = Path("merged/deduplication_report.json")
    
    if not input_csv.exists():
        print(f"❌ Error: {input_csv} not found!")
        return False
    
    try:
        # Load the dataset
        print(f"📂 Loading dataset from {input_csv}...")
        df = pd.read_csv(input_csv)
        
        print(f"✅ Loaded {len(df)} Q&A pairs")
        print(f"📊 Original dataset shape: {df.shape}")
        
        # Create deduplication report
        dedup_report = {
            "original_count": len(df),
            "deduplication_steps": []
        }
        
        # Step 1: Check for completely identical rows
        print(f"\n🔍 Step 1: Checking for completely identical rows...")
        identical_rows_before = len(df)
        df_step1 = df.drop_duplicates(keep='first')
        identical_duplicates = identical_rows_before - len(df_step1)
        
        dedup_report["deduplication_steps"].append({
            "step": "identical_rows",
            "removed": identical_duplicates,
            "remaining": len(df_step1),
            "description": "Removed completely identical rows across all columns"
        })
        
        if identical_duplicates > 0:
            print(f"  ✅ Removed {identical_duplicates} completely identical rows")
        else:
            print(f"  ✅ No completely identical rows found")
        
        # Step 2: Remove duplicate instructions (keep first occurrence)
        print(f"\n🔍 Step 2: Checking for duplicate instructions...")
        instruction_dupes_before = len(df_step1)
        df_step2 = df_step1.drop_duplicates(subset=['instruction'], keep='first')
        instruction_duplicates = instruction_dupes_before - len(df_step2)
        
        dedup_report["deduplication_steps"].append({
            "step": "duplicate_instructions",
            "removed": instruction_duplicates,
            "remaining": len(df_step2),
            "description": "Removed duplicate instructions (same question)"
        })
        
        if instruction_duplicates > 0:
            print(f"  ✅ Removed {instruction_duplicates} duplicate instructions")
            
            # Show some examples of removed duplicates
            duplicate_instructions = df_step1[df_step1.duplicated(subset=['instruction'], keep=False)]
            if len(duplicate_instructions) > 0:
                print(f"  📋 Examples of duplicate instructions removed:")
                unique_dupes = duplicate_instructions['instruction'].unique()[:3]
                for i, instruction in enumerate(unique_dupes, 1):
                    print(f"    {i}. \"{instruction[:80]}...\"")
        else:
            print(f"  ✅ No duplicate instructions found")
        
        # Step 3: Remove duplicate outputs (keep first occurrence)  
        print(f"\n🔍 Step 3: Checking for duplicate outputs...")
        output_dupes_before = len(df_step2)
        df_step3 = df_step2.drop_duplicates(subset=['output'], keep='first')
        output_duplicates = output_dupes_before - len(df_step3)
        
        dedup_report["deduplication_steps"].append({
            "step": "duplicate_outputs",
            "removed": output_duplicates,
            "remaining": len(df_step3),
            "description": "Removed duplicate outputs (same answer)"
        })
        
        if output_duplicates > 0:
            print(f"  ✅ Removed {output_duplicates} duplicate outputs")
        else:
            print(f"  ✅ No duplicate outputs found")
        
        # Step 4: Remove duplicate instruction-output pairs
        print(f"\n🔍 Step 4: Checking for duplicate instruction-output pairs...")
        qa_pairs_before = len(df_step3)
        df_final = df_step3.drop_duplicates(subset=['instruction', 'output'], keep='first')
        qa_pair_duplicates = qa_pairs_before - len(df_final)
        
        dedup_report["deduplication_steps"].append({
            "step": "duplicate_qa_pairs",
            "removed": qa_pair_duplicates,
            "remaining": len(df_final),
            "description": "Removed duplicate Q&A pairs (same question and answer)"
        })
        
        if qa_pair_duplicates > 0:
            print(f"  ✅ Removed {qa_pair_duplicates} duplicate Q&A pairs")
        else:
            print(f"  ✅ No duplicate Q&A pairs found")
        
        # Final statistics
        total_removed = len(df) - len(df_final)
        dedup_report["final_count"] = len(df_final)
        dedup_report["total_removed"] = total_removed
        dedup_report["removal_percentage"] = (total_removed / len(df)) * 100
        
        print(f"\n📊 Deduplication Summary:")
        print(f"Original samples: {len(df)}")
        print(f"Final samples: {len(df_final)}")
        print(f"Total removed: {total_removed}")
        print(f"Removal rate: {dedup_report['removal_percentage']:.1f}%")
        
        # Check data quality after deduplication
        print(f"\n🔍 Quality Check:")
        print(f"Unique instructions: {df_final['instruction'].nunique()}")
        print(f"Unique outputs: {df_final['output'].nunique()}")
        
        # Check source distribution
        if 'source_document' in df_final.columns:
            source_distribution = df_final['source_document'].value_counts()
            print(f"Sources represented: {len(source_distribution)}")
            print(f"Top 3 sources after deduplication:")
            for source, count in source_distribution.head(3).items():
                print(f"  - {source}: {count} pairs")
        
        # Save deduplicated CSV
        print(f"\n💾 Saving deduplicated dataset...")
        df_final.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✅ CSV saved: {output_csv}")
        
        # Save deduplicated JSON  
        df_final_records = df_final.to_dict('records')
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(df_final_records, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON saved: {output_json}")
        
        # Save deduplication report
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(dedup_report, f, indent=2, ensure_ascii=False)
        print(f"📋 Report saved: {report_file}")
        
        print(f"\n🎯 Ready for training!")
        print(f"Use the deduplicated dataset ({len(df_final)} samples) for fine-tuning.")
        
        return True, len(df_final)
        
    except Exception as e:
        print(f"❌ Error during deduplication: {e}")
        return False, 0

if __name__ == "__main__":
    print("🚀 Starting dataset deduplication...")
    print("=" * 60)
    
    success, final_count = deduplicate_dataset()
    
    print("=" * 60)
    if success:
        print("✅ Deduplication completed successfully!")
        print("\nFiles created:")
        print("  📄 merged/deduplicated_fraud_qa_dataset.csv")
        print("  📄 merged/deduplicated_fraud_qa_dataset.json") 
        print("  📋 merged/deduplication_report.json")
        print(f"\n🎯 Final dataset: {final_count} unique Q&A pairs")
        print("\nReady for fine-tuning with clean, deduplicated data!")
    else:
        print("❌ Deduplication failed!")