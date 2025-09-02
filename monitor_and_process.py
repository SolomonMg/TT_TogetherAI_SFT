#!/usr/bin/env python3
"""
monitor_and_process.py

Monitor the inference job progress and automatically run the pipeline when complete.
Checks for completion, then runs parse and merge steps automatically.

Usage:
    python monitor_and_process.py
"""

import os
import time
import subprocess
import json
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"[{time.strftime('%H:%M:%S')}] {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Success: {description}")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            print(f"✗ Failed: {description}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception in {description}: {e}")
        return False

def count_lines(file_path):
    """Count lines in a file."""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)
    except:
        return 0

def monitor_inference():
    """Monitor inference progress and return when complete."""
    target_file = "out/china_4.7_preds_remaining.raw.jsonl"
    expected_lines = 385028  # 395028 - 10000
    
    print(f"[{time.strftime('%H:%M:%S')}] Monitoring inference job...")
    print(f"Target file: {target_file}")
    print(f"Expected lines: {expected_lines:,}")
    
    last_count = 0
    stable_count = 0
    
    while True:
        if os.path.exists(target_file):
            current_count = count_lines(target_file)
            progress = (current_count / expected_lines) * 100 if expected_lines > 0 else 0
            
            print(f"[{time.strftime('%H:%M:%S')}] Progress: {current_count:,}/{expected_lines:,} ({progress:.1f}%)")
            
            # Check if file is complete (no new lines added for 10 minutes)
            if current_count == last_count:
                stable_count += 1
                if stable_count >= 10:  # 10 checks = ~10 minutes
                    print(f"[{time.strftime('%H:%M:%S')}] File appears complete (no new lines for 10 minutes)")
                    return current_count
            else:
                stable_count = 0
                last_count = current_count
                
            # Check if we've reached expected count
            if current_count >= expected_lines:
                print(f"[{time.strftime('%H:%M:%S')}] Reached expected line count!")
                return current_count
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for inference to start...")
        
        time.sleep(60)  # Check every minute

def run_pipeline():
    """Run the complete post-inference pipeline."""
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting post-inference pipeline...")
    
    # Step 1: Parse the remaining predictions
    parse_cmd = "python parse.py --raw out/china_4.7_preds_remaining.raw.jsonl --out out/china_4.7_preds_remaining.parsed.jsonl --print-bad 5"
    if not run_command(parse_cmd, "Parsing remaining predictions"):
        return False
    
    # Step 2: Combine both parsed files
    combine_cmd = """python -c "
import json

# Load both parsed files
with open('out/china_4.7_preds_10k.parsed.jsonl', 'r') as f:
    data_10k = [json.loads(line) for line in f if line.strip()]

with open('out/china_4.7_preds_remaining.parsed.jsonl', 'r') as f:
    data_remaining = [json.loads(line) for line in f if line.strip()]

# Combine and write
combined = data_10k + data_remaining
print(f'Combining {len(data_10k):,} + {len(data_remaining):,} = {len(combined):,} predictions')

with open('out/china_4.7_preds_all.parsed.jsonl', 'w') as f:
    for item in combined:
        f.write(json.dumps(item) + '\\n')

print('Wrote combined predictions to out/china_4.7_preds_all.parsed.jsonl')
"
"""
    if not run_command(combine_cmd, "Combining parsed prediction files"):
        return False
    
    # Step 3: Merge all predictions with original data
    merge_cmd = """python -c "
from json_utils import merge_predictions_with_csv
merge_predictions_with_csv(
    csv_path='data/china_4.7_sample_all_with_caption.parquet',
    parsed_jsonl_path='out/china_4.7_preds_all.parsed.jsonl',
    output_path='out/china_4.7_with_all_predictions.csv'
)
"
"""
    if not run_command(merge_cmd, "Merging all predictions with original data"):
        return False
    
    # Step 4: Generate summary statistics
    summary_cmd = """python -c "
import pandas as pd
import json

print('\\n=== FINAL SUMMARY ===')

# Load final merged data
df = pd.read_csv('out/china_4.7_with_all_predictions.csv')
predicted_rows = df[df['pred_parsed'] == True]

print(f'Total rows in dataset: {len(df):,}')
print(f'Rows with predictions: {len(predicted_rows):,} ({len(predicted_rows)/len(df)*100:.1f}%)')

if len(predicted_rows) > 0:
    print('\\nStance Score Distribution:')
    scores = predicted_rows['pred_china_stance_score'].dropna()
    if len(scores) > 0:
        print(f'  Count: {len(scores):,}')
        print(f'  Mean: {scores.mean():.3f}')
        print(f'  Min: {scores.min():.3f}')
        print(f'  Max: {scores.max():.3f}')
        
        negative = len(scores[scores < -0.1])
        neutral = len(scores[(scores >= -0.1) & (scores <= 0.1)])
        positive = len(scores[scores > 0.1])
        
        print(f'  Negative (< -0.1): {negative:,} ({negative/len(scores)*100:.1f}%)')
        print(f'  Neutral (-0.1 to 0.1): {neutral:,} ({neutral/len(scores)*100:.1f}%)')
        print(f'  Positive (> 0.1): {positive:,} ({positive/len(scores)*100:.1f}%)')
    
    print('\\nSensitivity Distribution:')
    sens_counts = predicted_rows['pred_china_sensitive'].value_counts()
    for val, count in sens_counts.items():
        pct = count/len(predicted_rows)*100
        print(f'  {val}: {count:,} ({pct:.1f}%)')

print(f'\\nFinal output: out/china_4.7_with_all_predictions.csv')
"
"""
    if not run_command(summary_cmd, "Generating final summary"):
        return False
    
    print(f"\n[{time.strftime('%H:%M:%S')}] ✓ Pipeline complete!")
    return True

def main():
    print("=== China 4.7 Inference Pipeline Monitor ===")
    print(f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Monitor inference until complete
    final_count = monitor_inference()
    print(f"[{time.strftime('%H:%M:%S')}] Inference complete with {final_count:,} examples")
    
    # Run the post-processing pipeline
    if run_pipeline():
        print(f"\n🎉 All processing complete at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\n❌ Pipeline failed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()