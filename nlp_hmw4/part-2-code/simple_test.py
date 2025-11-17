#!/usr/bin/env python3
"""Simple test script"""
try:
    from load_data import load_t5_data
    print("✓ Import load_data successful")
    
    train_loader, dev_loader, test_loader = load_t5_data(batch_size=2, test_batch_size=2)
    print(f"✓ Data loading successful")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Dev batches: {len(dev_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    batch = next(iter(train_loader))
    print(f"✓ Batch retrieval successful")
    print(f"  Batch has {len(batch)} elements")
    
    from t5_utils import initialize_model
    import argparse
    args = argparse.Namespace()
    args.finetune = True
    model = initialize_model(args)
    print(f"✓ Model initialization successful")
    
    print("ALL TESTS PASSED")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

