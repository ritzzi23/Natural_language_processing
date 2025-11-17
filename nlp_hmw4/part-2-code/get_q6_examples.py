#!/usr/bin/env python3
"""
Get specific examples for Q6 error analysis
"""
import pickle
from utils import read_queries, compute_metrics

def get_error_examples():
    # Load data
    gt_queries = read_queries('data/dev.sql')
    pred_queries = read_queries('results/t5_ft_ft_experiment_dev.sql')
    
    with open('records/ground_truth_dev.pkl', 'rb') as f:
        gt_records, _ = pickle.load(f)
    
    with open('records/t5_ft_ft_experiment_dev.pkl', 'rb') as f:
        pred_records, error_msgs = pickle.load(f)
    
    with open('data/dev.nl', 'r') as f:
        nl_queries = [line.strip() for line in f.readlines()]
    
    # Find examples for each error type
    errors = {
        'missing_condition': [],
        'wrong_column': [],
        'wrong_aggregation': [],
    }
    
    for i, (gt_q, pred_q, nl_q, gt_rec, pred_rec) in enumerate(zip(
        gt_queries, pred_queries, nl_queries, gt_records, pred_records
    )):
        # Only look at cases where records don't match
        if gt_rec == pred_rec:
            continue
        
        # Missing condition (WHERE clause missing or incomplete)
        if 'WHERE' in gt_q.upper() and 'WHERE' not in pred_q.upper():
            errors['missing_condition'].append({
                'nl': nl_q,
                'gt': gt_q,
                'pred': pred_q,
            })
        elif 'WHERE' in gt_q.upper() and 'WHERE' in pred_q.upper():
            # Check if conditions are different
            gt_where = gt_q.upper().split('WHERE')[1].split('ORDER')[0].split('GROUP')[0]
            pred_where = pred_q.upper().split('WHERE')[1].split('ORDER')[0].split('GROUP')[0]
            if gt_where != pred_where and len(errors['missing_condition']) < 3:
                errors['missing_condition'].append({
                    'nl': nl_q,
                    'gt': gt_q,
                    'pred': pred_q,
                })
        
        # Wrong column selection
        gt_select = gt_q.upper().split('SELECT')[1].split('FROM')[0].strip()
        pred_select = pred_q.upper().split('SELECT')[1].split('FROM')[0].strip()
        if gt_select != pred_select and len(errors['wrong_column']) < 3:
            errors['wrong_column'].append({
                'nl': nl_q,
                'gt': gt_q,
                'pred': pred_q,
            })
        
        # Wrong aggregation
        if 'MIN(' in gt_q.upper() or 'MAX(' in gt_q.upper() or 'COUNT(' in gt_q.upper():
            if ('MIN(' in gt_q.upper() and 'MIN(' not in pred_q.upper()) or \
               ('MAX(' in gt_q.upper() and 'MAX(' not in pred_q.upper()) or \
               ('COUNT(' in gt_q.upper() and 'COUNT(' not in pred_q.upper()):
                if len(errors['wrong_aggregation']) < 3:
                    errors['wrong_aggregation'].append({
                        'nl': nl_q,
                        'gt': gt_q,
                        'pred': pred_q,
                    })
    
    return errors

if __name__ == "__main__":
    errors = get_error_examples()
    
    print("=" * 80)
    print("ERROR EXAMPLES FOR TABLE 5")
    print("=" * 80)
    
    for error_type, examples in errors.items():
        if len(examples) > 0:
            print(f"\n{error_type.replace('_', ' ').title()}:")
            for i, ex in enumerate(examples[:2]):
                print(f"\n  Example {i+1}:")
                print(f"    NL: {ex['nl']}")
                print(f"    GT: {ex['gt']}")
                print(f"    Pred: {ex['pred']}")

