#!/usr/bin/env python3
"""
Detailed error analysis for Q6
"""
import pickle
import re
from utils import read_queries, compute_metrics

def extract_sql_components(query):
    """Extract key components from SQL query"""
    query_upper = query.upper()
    return {
        'tables': set(re.findall(r'\bFROM\s+(\w+)', query_upper)),
        'select_cols': re.findall(r'SELECT\s+(.*?)\s+FROM', query_upper),
        'where_clauses': re.findall(r'WHERE\s+(.*?)(?:\s+GROUP|\s+ORDER|\s+UNION|$)', query_upper, re.DOTALL),
        'joins': len(re.findall(r'\bJOIN\b', query_upper)),
        'aggregations': set(re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', query_upper)),
        'order_by': 'ORDER BY' in query_upper,
        'distinct': 'DISTINCT' in query_upper,
    }

def analyze_semantic_errors(gt_queries, pred_queries, nl_queries, gt_records, pred_records):
    """Analyze semantic differences between GT and predictions"""
    
    errors = {
        'wrong_column_selection': [],
        'missing_or_wrong_condition': [],
        'incorrect_join_structure': [],
        'wrong_aggregation': [],
        'missing_distinct': [],
        'wrong_ordering': [],
    }
    
    for i, (gt_q, pred_q, nl_q, gt_rec, pred_rec) in enumerate(zip(
        gt_queries, pred_queries, nl_queries, gt_records, pred_records
    )):
        # Skip if records match exactly
        if gt_rec == pred_rec:
            continue
        
        gt_comp = extract_sql_components(gt_q)
        pred_comp = extract_sql_components(pred_q)
        
        # Wrong column selection
        if gt_comp['select_cols'] != pred_comp['select_cols']:
            errors['wrong_column_selection'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q[:200],
                'pred': pred_q[:200],
            })
        
        # Missing or wrong conditions
        if gt_comp['where_clauses'] and not pred_comp['where_clauses']:
            errors['missing_or_wrong_condition'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q[:200],
                'pred': pred_q[:200],
            })
        elif gt_comp['where_clauses'] and pred_comp['where_clauses']:
            # Check if conditions are different
            gt_where = ' '.join(gt_comp['where_clauses']).upper()
            pred_where = ' '.join(pred_comp['where_clauses']).upper()
            if gt_where != pred_where:
                errors['missing_or_wrong_condition'].append({
                    'idx': i,
                    'nl': nl_q,
                    'gt': gt_q[:200],
                    'pred': pred_q[:200],
                })
        
        # Incorrect join structure
        if gt_comp['joins'] != pred_comp['joins']:
            errors['incorrect_join_structure'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q[:200],
                'pred': pred_q[:200],
            })
        
        # Wrong aggregation
        if gt_comp['aggregations'] != pred_comp['aggregations']:
            errors['wrong_aggregation'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q[:200],
                'pred': pred_q[:200],
            })
        
        # Missing DISTINCT
        if gt_comp['distinct'] and not pred_comp['distinct']:
            errors['missing_distinct'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q[:200],
                'pred': pred_q[:200],
            })
        
        # Wrong ordering
        if gt_comp['order_by'] != pred_comp['order_by']:
            errors['wrong_ordering'].append({
                'idx': i,
                'nl': nl_q,
                'gt': gt_q[:200],
                'pred': pred_q[:200],
            })
    
    return errors

def main():
    # Load data
    gt_queries = read_queries('data/dev.sql')
    pred_queries = read_queries('results/t5_ft_ft_experiment_dev.sql')
    
    with open('records/ground_truth_dev.pkl', 'rb') as f:
        gt_records, _ = pickle.load(f)
    
    with open('records/t5_ft_ft_experiment_dev.pkl', 'rb') as f:
        pred_records, error_msgs = pickle.load(f)
    
    with open('data/dev.nl', 'r') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    # Compute metrics
    sql_em, record_em, record_f1, _ = compute_metrics(
        'data/dev.sql',
        'results/t5_ft_ft_experiment_dev.sql',
        'records/ground_truth_dev.pkl',
        'records/t5_ft_ft_experiment_dev.pkl'
    )
    
    print("=" * 80)
    print("QUANTITATIVE RESULTS")
    print("=" * 80)
    print(f"\nDevelopment Set:")
    print(f"  SQL Query EM: {sql_em*100:.2f}%")
    print(f"  Record EM: {record_em*100:.2f}%")
    print(f"  Record F1: {record_f1*100:.2f}%")
    
    # Count SQL execution errors
    sql_errors = sum(1 for msg in error_msgs if msg is not None)
    print(f"\nSQL Execution Errors: {sql_errors}/{len(gt_queries)} ({sql_errors/len(gt_queries)*100:.1f}%)")
    
    # Analyze errors
    print("\n" + "=" * 80)
    print("QUALITATIVE ERROR ANALYSIS")
    print("=" * 80)
    
    errors = analyze_semantic_errors(gt_queries, pred_queries, nl_queries, gt_records, pred_records)
    
    # Count total incorrect predictions
    incorrect = sum(1 for gt_rec, pred_rec in zip(gt_records, pred_records) if gt_rec != pred_rec)
    
    print(f"\nTotal incorrect predictions: {incorrect}/{len(gt_queries)} ({incorrect/len(gt_queries)*100:.1f}%)")
    print(f"Correct predictions: {len(gt_queries) - incorrect}/{len(gt_queries)} ({(len(gt_queries) - incorrect)/len(gt_queries)*100:.1f}%)")
    
    # Print top error types
    print("\n" + "-" * 80)
    print("ERROR TYPE STATISTICS")
    print("-" * 80)
    
    error_counts = {k: len(v) for k, v in errors.items() if len(v) > 0}
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    for error_type, count in sorted_errors[:6]:  # Top 6 error types
        examples = errors[error_type]
        print(f"\n{error_type.replace('_', ' ').title()}: {count}/{incorrect} ({count/incorrect*100:.1f}% of errors)")
        if len(examples) > 0:
            ex = examples[0]
            print(f"  Example NL: {ex['nl'][:120]}")
            print(f"  Example GT: {ex['gt']}")
            print(f"  Example Pred: {ex['pred']}")
    
    # Get specific examples for report
    print("\n" + "=" * 80)
    print("EXAMPLES FOR TABLE 5")
    print("=" * 80)
    
    # Get 3 distinct error types with examples
    selected_errors = []
    for error_type, examples in errors.items():
        if len(examples) > 0 and len(selected_errors) < 3:
            selected_errors.append((error_type, examples))
    
    for error_type, examples in selected_errors:
        print(f"\n{error_type.replace('_', ' ').title()}:")
        for i, ex in enumerate(examples[:2]):  # Show 2 examples
            print(f"  Example {i+1}:")
            print(f"    NL: {ex['nl'][:100]}...")
            print(f"    GT: {ex['gt']}")
            print(f"    Pred: {ex['pred']}")

if __name__ == "__main__":
    main()

