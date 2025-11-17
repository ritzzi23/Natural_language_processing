#!/usr/bin/env python3
"""
Verify that all important files are present locally
"""
import os
from pathlib import Path

# Key files that should be present
required_files = {
    'results': [
        't5_ft_ft_experiment_dev.sql',
        't5_ft_ft_experiment_test.sql',
    ],
    'records': [
        't5_ft_ft_experiment_dev.pkl',
        't5_ft_ft_experiment_test.pkl',
        'ground_truth_dev.pkl',
    ],
    'checkpoints': [
        'ft_experiments/baseline/best_model.pt',
        'ft_experiments/baseline/checkpoint.pt',
    ],
    'root': [
        'training.log',
    ]
}

print("=" * 80)
print("FILE VERIFICATION CHECK")
print("=" * 80)

all_present = True
missing_files = []

for category, files in required_files.items():
    print(f"\n{category.upper()}:")
    for file in files:
        if category == 'root':
            path = file
        else:
            path = os.path.join(category, file)
        
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_mb = size / (1024 * 1024)
            print(f"  ✓ {path} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {path} - MISSING")
            missing_files.append(path)
            all_present = False

print("\n" + "=" * 80)
if all_present:
    print("✓ ALL REQUIRED FILES ARE PRESENT")
else:
    print(f"✗ MISSING {len(missing_files)} FILE(S):")
    for f in missing_files:
        print(f"  - {f}")

print("\n" + "=" * 80)
print("ADDITIONAL FILES FOUND:")
print("=" * 80)

# Check for additional files
additional = {
    'results': [f for f in os.listdir('results') if f.endswith('.sql')],
    'records': [f for f in os.listdir('records') if f.endswith('.pkl')],
}

for category, files in additional.items():
    if files:
        print(f"\n{category.upper()}:")
        for f in sorted(files):
            path = os.path.join(category, f)
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")

