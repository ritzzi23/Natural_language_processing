#!/usr/bin/env python3
"""
Analyze Q6 Results: Compute metrics and identify error types
"""
import pickle
import sqlite3
from utils import compute_metrics, read_queries
from collections import defaultdict
import re

def analyze_errors(gt_sql_path, pred_sql_path, gt_records_path, pred_records_path, dev_nl_path):
    """
    Analyze errors in predictions and categorize them
    """
    # Load data
    gt_queries = read_queries(gt_sql_path)
    pred_queries = read_queries(pred_sql_path)
    
    with open(gt_records_path, 'rb') as f:
        gt_records, _ = pickle.load(f)
    
    with open(pred_records_path, 'rb') as f:
        pred_records, error_msgs = pickle.load(f)
    
    # Load NL queries for examples
    with open(dev_nl_path, 'r') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    # Compute metrics
    sql_em, record_em, record_f1, _ = compute_metrics(
        gt_sql_path, pred_sql_path, gt_records_path, pred_records_path
    )
    
    print("=" * 80)
    print("QUANTITATIVE RESULTS")
    print("=" * 80)
    print(f"\nDevelopment Set Results:")
    print(f"  SQL Query EM: {sql_em*100:.2f}%")
    print(f"  Record EM: {record_em*100:.2f}%")
    print(f"  Record F1: {record_f1*100:.2f}%")
    print(f"\nTotal examples: {len(gt_queries)}")
    print(f"SQL errors: {sum(1 for msg in error_msgs if msg is not None)}")
    
    # Analyze error types
    print("\n" + "=" * 80)
    print("QUALITATIVE ERROR ANALYSIS")
    print("=" * 80)
    
    errors = {
        'sql_syntax_error': [],
        'wrong_table_column': [],
        'missing_condition': [],
        'wrong_aggregation': [],
        'incorrect_join': [],
        'wrong_ordering': [],
    }
    
    # Categorize errors
    for i, (gt_q, pred_q, gt_rec, pred_rec, nl_q, err_msg) in enumerate(zip(
        gt_queries, pred_queries, gt_records, pred_records, nl_queries, error_msgs
    )):
        # Skip if SQL query matches exactly
        if gt_q.strip().upper() == pred_q.strip().upper():
            continue
        
        # SQL syntax errors
        if err_msg is not None:
            errors['sql_syntax_error'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q,
                'pred': pred_q,
                'error': err_msg
            })
            continue
        
        # Wrong table/column names
        gt_tables = set(re.findall(r'\bFROM\s+(\w+)', gt_q.upper()))
        pred_tables = set(re.findall(r'\bFROM\s+(\w+)', pred_q.upper()))
        gt_cols = set(re.findall(r'\b(\w+\.\w+|\w+)\s*=', gt_q.upper()))
        pred_cols = set(re.findall(r'\b(\w+\.\w+|\w+)\s*=', pred_q.upper()))
        
        if gt_tables != pred_tables or len(gt_cols - pred_cols) > 2:
            errors['wrong_table_column'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q,
                'pred': pred_q,
            })
            continue
        
        # Missing conditions (WHERE clauses)
        gt_where = 'WHERE' in gt_q.upper()
        pred_where = 'WHERE' in pred_q.upper()
        if gt_where and not pred_where:
            errors['missing_condition'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q,
                'pred': pred_q,
            })
            continue
        
        # Wrong aggregation (COUNT, SUM, etc.)
        gt_agg = set(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', gt_q.upper()))
        pred_agg = set(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', pred_q.upper()))
        if gt_agg != pred_agg and len(gt_agg) > 0:
            errors['wrong_aggregation'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q,
                'pred': pred_q,
            })
            continue
        
        # Incorrect JOINs
        gt_joins = len(re.findall(r'\bJOIN\b', gt_q.upper()))
        pred_joins = len(re.findall(r'\bJOIN\b', pred_q.upper()))
        if gt_joins != pred_joins:
            errors['incorrect_join'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q,
                'pred': pred_q,
            })
            continue
        
        # Wrong ordering (ORDER BY)
        gt_order = 'ORDER BY' in gt_q.upper()
        pred_order = 'ORDER BY' in pred_q.upper()
        if gt_order != pred_order:
            errors['wrong_ordering'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q,
                'pred': pred_q,
            })
            continue
    
    # Print error statistics
    print("\nError Type Statistics:")
    print("-" * 80)
    total_errors = len(gt_queries) - sum(1 for i, (g, p) in enumerate(zip(gt_queries, pred_queries)) 
                                         if g.strip().upper() == p.strip().upper())
    
    for error_type, examples in errors.items():
        if len(examples) > 0:
            print(f"\n{error_type.replace('_', ' ').title()}: {len(examples)}/{len(gt_queries)} ({len(examples)/len(gt_queries)*100:.1f}%)")
            if len(examples) > 0:
                ex = examples[0]
                print(f"  Example NL: {ex['nl'][:100]}...")
                print(f"  Example GT: {ex['gt'][:150]}...")
                print(f"  Example Pred: {ex['pred'][:150]}...")
    
    return {
        'sql_em': sql_em,
        'record_em': record_em,
        'record_f1': record_f1,
        'errors': errors,
        'total_examples': len(gt_queries)
    }

if __name__ == "__main__":
    # Dev set analysis
    print("Analyzing Development Set...")
    dev_results = analyze_errors(
        'data/dev.sql',
        'results/t5_ft_ft_experiment_dev.sql',
        'records/ground_truth_dev.pkl',
        'records/t5_ft_ft_experiment_dev.pkl',
        'data/dev.nl'
    )
    
    print("\n" + "=" * 80)
    print("SUMMARY FOR TABLE 4")
    print("=" * 80)
    print(f"\nDev Results:")
    print(f"  SQL Query EM: {dev_results['sql_em']*100:.2f}%")
    print(f"  Record F1: {dev_results['record_f1']*100:.2f}%")
    
    # Test set (if available)
    try:
        test_results = compute_metrics(
            'data/test.sql',
            'results/t5_ft_ft_experiment_test.sql',
            None,  # No GT records for test
            'records/t5_ft_ft_experiment_test.pkl'
        )
        print(f"\nTest Results:")
        print(f"  SQL Query EM: {test_results[0]*100:.2f}%")
        print(f"  Record F1: {test_results[2]*100:.2f}%")
    except Exception as e:
        print(f"\nTest results not available: {e}")

