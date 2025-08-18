#!/usr/bin/env python3
"""
Merge all final_combined_qa_pairs.json files from data_sources_v1, data_sources_v2, data_sources_v3
into a single consolidated dataset in the merged folder.
"""

import json
import os
import glob
from pathlib import Path

def find_qa_files():
    """Find all final_combined_qa_pairs.json files in data_sources directories"""
    
    qa_files = []
    base_dir = Path(".")
    
    # Look for data_sources, data_sources_v1, data_sources_v2, data_sources_v3
    patterns = [
        "data_sources/*/processed/final_combined_qa_pairs.json",
        "data_sources_v1/*/processed/final_combined_qa_pairs.json", 
        "data_sources_v2/*/processed/final_combined_qa_pairs.json",
        "data_sources_v3/*/processed/final_combined_qa_pairs.json"
    ]
    
    for pattern in patterns:
        files = glob.glob(str(base_dir / pattern))
        qa_files.extend(files)
    
    return qa_files

def load_json_file(file_path):
    """Load and validate JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print(f"✅ Loaded {len(data)} Q&A pairs from {file_path}")
            return data
        else:
            print(f"⚠️  Warning: {file_path} does not contain a list")
            return []
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error in {file_path}: {e}")
        return []
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

def merge_qa_datasets():
    """Main function to merge all QA datasets"""
    
    print("🔍 Searching for QA files...")
    qa_files = find_qa_files()
    
    if not qa_files:
        print("❌ No final_combined_qa_pairs.json files found!")
        print("Expected locations:")
        print("  - data_sources/*/processed/final_combined_qa_pairs.json")
        print("  - data_sources_v1/*/processed/final_combined_qa_pairs.json")
        print("  - data_sources_v2/*/processed/final_combined_qa_pairs.json") 
        print("  - data_sources_v3/*/processed/final_combined_qa_pairs.json")
        return
    
    print(f"📁 Found {len(qa_files)} QA files:")
    for file_path in qa_files:
        print(f"  - {file_path}")
    
    # Load all data
    all_qa_pairs = []
    source_stats = {}
    
    for file_path in qa_files:
        data = load_json_file(file_path)
        if data:
            all_qa_pairs.extend(data)
            
            # Track source statistics
            source_name = Path(file_path).parts[-3]  # Get source name from path
            source_stats[source_name] = len(data)
    
    if not all_qa_pairs:
        print("❌ No valid QA pairs found in any files!")
        return
    
    print(f"\n📊 Dataset Statistics:")
    print(f"Total Q&A pairs: {len(all_qa_pairs)}")
    print("Source breakdown:")
    for source, count in source_stats.items():
        print(f"  - {source}: {count} pairs")
    
    # Create merged directory
    merged_dir = Path("merged")
    merged_dir.mkdir(exist_ok=True)
    print(f"\n📁 Created/verified merged directory: {merged_dir}")
    
    # Save merged dataset
    output_file = merged_dir / "consolidated_fraud_qa_dataset.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully saved merged dataset to: {output_file}")
        print(f"📈 Total samples in merged dataset: {len(all_qa_pairs)}")
        
        # Create summary report
        summary = {
            "total_samples": len(all_qa_pairs),
            "source_breakdown": source_stats,
            "source_files": qa_files,
            "output_file": str(output_file)
        }
        
        summary_file = merged_dir / "merge_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Summary report saved to: {summary_file}")
        
        # Sample validation
        if all_qa_pairs:
            sample = all_qa_pairs[0]
            print(f"\n🔍 Sample Q&A pair structure:")
            print(f"Keys: {list(sample.keys())}")
            if 'instruction' in sample:
                print(f"Sample instruction: {sample['instruction'][:100]}...")
        
        return output_file, len(all_qa_pairs)
        
    except Exception as e:
        print(f"❌ Error saving merged dataset: {e}")
        return None, 0

if __name__ == "__main__":
    print("🚀 Starting QA dataset merge process...")
    print("=" * 50)
    
    result_file, total_samples = merge_qa_datasets()
    
    print("=" * 50)
    if result_file:
        print(f"✅ Merge completed successfully!")
        print(f"📄 Output: {result_file}")
        print(f"📊 Total samples: {total_samples}")
    else:
        print("❌ Merge failed!")