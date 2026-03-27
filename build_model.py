#!/usr/bin/env python3
"""
Apply signals to transcript corpus, build scoring model, validate, extract citations.

Library entry point:
    build_signal_model(output_dir, signals_file)

Standalone:
    python3 build_model.py --output-dir ./output --signals-file signals.json

Reads:
    {output_dir}/transcript_corpus.json
    signals_file (JSON array of signal definitions)

Outputs (in output_dir):
    final_model.json   — full model with signals, weights, metrics, citations
    citations.json     — per-signal citation data
"""

import json
import re
import sys
import random
import argparse
from collections import defaultdict
from statistics import mean


def load_signals(signals_file):
    """Load and validate signals from a JSON file.

    Supports two formats:
    - Regex format: requires name, description, patterns, direction, source_meeting
    - LLM format: requires name, description, direction, category (no patterns)

    Format is auto-detected by checking if the first signal has a 'patterns' key.
    """
    with open(signals_file, 'r') as f:
        signals = json.load(f)

    if not isinstance(signals, list):
        print(f"ERROR: Signals file must contain a JSON array, got {type(signals).__name__}")
        sys.exit(1)

    if not signals:
        print("ERROR: Signals file is empty")
        sys.exit(1)

    # Detect format
    is_regex_format = 'patterns' in signals[0]

    if is_regex_format:
        required_keys = {'name', 'description', 'patterns', 'direction', 'source_meeting'}
        for i, sig in enumerate(signals):
            missing = required_keys - set(sig.keys())
            if missing:
                print(f"ERROR: Signal {i} missing keys: {missing}")
                sys.exit(1)
            if sig['direction'] not in ('positive', 'negative'):
                print(f"ERROR: Signal {i} ({sig['name']}): direction must be 'positive' or 'negative', got '{sig['direction']}'")
                sys.exit(1)
            if not isinstance(sig['patterns'], list) or not sig['patterns']:
                print(f"ERROR: Signal {i} ({sig['name']}): patterns must be a non-empty list")
                sys.exit(1)
            for j, pat in enumerate(sig['patterns']):
                try:
                    re.compile(pat)
                except re.error as e:
                    print(f"ERROR: Signal {i} ({sig['name']}): invalid regex in pattern {j}: {e}")
                    sys.exit(1)
        print(f"  Signal format: regex ({len(signals)} signals)")
    else:
        required_keys = {'name', 'description', 'direction', 'category'}
        for i, sig in enumerate(signals):
            missing = required_keys - set(sig.keys())
            if missing:
                print(f"ERROR: Signal {i} missing keys: {missing}")
                sys.exit(1)
            if sig['direction'] not in ('positive', 'negative'):
                print(f"ERROR: Signal {i} ({sig['name']}): direction must be 'positive' or 'negative', got '{sig['direction']}'")
                sys.exit(1)
        print(f"  Signal format: LLM ({len(signals)} signals)")

    return signals


def signal_fires(text_lower, signal):
    """Check if any of the signal's patterns match in the text."""
    for pat in signal["patterns"]:
        if re.search(pat, text_lower):
            return True
    return False


def find_citation(text, signal, max_context=150):
    """Find the first matching snippet for a signal in the text."""
    text_lower = text.lower()
    for pat in signal["patterns"]:
        m = re.search(pat, text_lower)
        if m:
            start = max(0, m.start() - max_context)
            end = min(len(text), m.end() + max_context)
            snippet = text[start:end].strip()
            if start > 0:
                first_space = snippet.find(' ')
                if first_space > 0:
                    snippet = "..." + snippet[first_space:]
            if end < len(text):
                last_space = snippet.rfind(' ')
                if last_space > 0 and last_space < len(snippet) - 1:
                    snippet = snippet[:last_space] + "..."
            return snippet
    return None


def _load_evaluations(evaluations_file, signals):
    """Load and validate pre-computed signal evaluations.

    Returns evaluations dict keyed by opp_name -> signal_name -> {fires, citation}.
    """
    import hashlib

    with open(evaluations_file) as f:
        eval_data = json.load(f)

    # Validate signals hash
    signals_hash = hashlib.sha256(json.dumps(signals, sort_keys=True).encode()).hexdigest()
    file_hash = eval_data.get('metadata', {}).get('signals_hash', '')
    if file_hash != signals_hash:
        print(f"WARNING: Evaluations signals_hash mismatch!")
        print(f"  Evaluations file: {file_hash[:16]}...")
        print(f"  Current signals:  {signals_hash[:16]}...")
        print(f"  Results may be inconsistent. Re-run --generate-signals to fix.")

    evaluations = eval_data.get('evaluations', {})
    print(f"  Loaded evaluations for {len(evaluations)} opportunities")
    return evaluations


def build_signal_model(output_dir, signals_file, evaluations_file=None):
    """Library entry point: build and validate signal model.

    Args:
        output_dir: Directory containing transcript_corpus.json and for output.
        signals_file: Path to signals JSON file.
        evaluations_file: Optional path to pre-computed signal evaluations JSON.
            If provided, uses LLM evaluations instead of regex matching.

    Returns:
        dict summary with model performance metrics.
    """
    signals = load_signals(signals_file)
    print(f"Loaded {len(signals)} signals from {signals_file}")

    use_evaluations = evaluations_file is not None
    eval_lookup = None
    if use_evaluations:
        print(f"Loading evaluations from {evaluations_file}...")
        eval_lookup = _load_evaluations(evaluations_file, signals)

    # Load corpus
    corpus_path = f"{output_dir}/transcript_corpus.json"
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path) as f:
        corpus = json.load(f)

    won_opps = corpus['won']
    lost_opps = corpus['lost']
    all_opps = {}
    for name, data in won_opps.items():
        all_opps[name] = {**data, 'cohort': 'won'}
    for name, data in lost_opps.items():
        all_opps[name] = {**data, 'cohort': 'lost'}

    print(f"Loaded {len(won_opps)} Won, {len(lost_opps)} Lost opportunities")

    # Apply signals to all opps
    print("\nApplying signals...")
    signal_results = {}
    if use_evaluations:
        for opp_name in all_opps:
            signal_results[opp_name] = {}
            opp_eval = eval_lookup.get(opp_name, {})
            for sig in signals:
                entry = opp_eval.get(sig['name'], {})
                signal_results[opp_name][sig['name']] = bool(entry.get('fires', False))
        print(f"  Applied from pre-computed evaluations")
    else:
        for opp_name, opp_data in all_opps.items():
            text_lower = opp_data['full_text'].lower()
            signal_results[opp_name] = {}
            for sig in signals:
                signal_results[opp_name][sig['name']] = signal_fires(text_lower, sig)

    # Full-dataset prevalence
    print("\n=== FULL DATASET PREVALENCE ===")
    signal_prevalence = {}
    for sig in signals:
        won_fires = sum(1 for name in won_opps if signal_results[name][sig['name']])
        lost_fires = sum(1 for name in lost_opps if signal_results[name][sig['name']])
        won_pct = won_fires / len(won_opps) * 100
        lost_pct = lost_fires / len(lost_opps) * 100
        delta = won_pct - lost_pct
        total_pct = (won_fires + lost_fires) / len(all_opps) * 100

        signal_prevalence[sig['name']] = {
            'won_fires': won_fires, 'lost_fires': lost_fires,
            'won_pct': round(won_pct, 1), 'lost_pct': round(lost_pct, 1),
            'delta': round(delta, 1), 'total_pct': round(total_pct, 1),
        }
        print(f"  {sig['name']:45s} Won:{won_pct:6.1f}% Lost:{lost_pct:6.1f}% "
              f"Delta:{delta:+6.1f}pp [{won_fires}/{len(won_opps)} vs {lost_fires}/{len(lost_opps)}]")

    # Train/Test split (80/20 deterministic by sorted name)
    won_sorted = sorted(won_opps.keys())
    lost_sorted = sorted(lost_opps.keys())
    won_train_size = int(len(won_sorted) * 0.8)
    lost_train_size = int(len(lost_sorted) * 0.8)

    won_train = set(won_sorted[:won_train_size])
    won_test = set(won_sorted[won_train_size:])
    lost_train = set(lost_sorted[:lost_train_size])
    lost_test = set(lost_sorted[lost_train_size:])

    print(f"\nSplit: Won train={len(won_train)}, test={len(won_test)}; "
          f"Lost train={len(lost_train)}, test={len(lost_test)}")

    # Classify signals into tiers
    tier1 = []  # Variable (|delta| > 0)
    tier2 = []  # Universal (delta = 0)
    for sig in signals:
        d = signal_prevalence[sig['name']]['delta']
        if abs(d) > 0:
            tier1.append(sig)
        else:
            tier2.append(sig)

    print(f"\nTier 1 (variable, |delta|>0): {len(tier1)} signals")
    print(f"Tier 2 (universal, delta=0): {len(tier2)} signals")

    # All signals in the model; variable get weight=delta/100, universal get 0
    all_model_signals = tier1 + tier2
    signal_weights = {}
    for sig in all_model_signals:
        d = signal_prevalence[sig['name']]['delta']
        signal_weights[sig['name']] = d / 100.0

    # Per-signal precision/recall on TEST set
    print("\n=== PER-SIGNAL TEST METRICS ===")
    signal_metrics = {}
    test_opps_list = sorted(list(won_test) + list(lost_test))

    for sig in all_model_signals:
        name = sig['name']
        direction = sig['direction']

        tp = fp = fn = tn = 0
        for opp_name in test_opps_list:
            fires = signal_results[opp_name][name]
            cohort = all_opps[opp_name]['cohort']
            is_correct = (cohort == 'won') if direction == 'positive' else (cohort == 'lost')

            if fires and is_correct:
                tp += 1
            elif fires and not is_correct:
                fp += 1
            elif not fires and is_correct:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        signal_metrics[name] = {
            'precision': round(precision, 3), 'recall': round(recall, 3),
            'f1': round(f1, 3), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }
        print(f"  {name:45s} P:{precision:.3f} R:{recall:.3f} F1:{f1:.3f}")

    # Monte Carlo validation
    print("\n=== MONTE CARLO VALIDATION (TEST SET, 1000 trials) ===")
    rng = random.Random(42)
    n_trials = 1000
    mc_data = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for trial in range(n_trials):
        t_tp = t_fp = t_fn = t_tn = 0
        for opp_name in test_opps_list:
            cohort = all_opps[opp_name]['cohort']
            score = 0.0
            for sig in all_model_signals:
                fires = signal_results[opp_name][sig['name']]
                if rng.random() < 0.05:
                    fires = not fires
                if fires:
                    score += signal_weights[sig['name']]

            predicted_won = score > 0
            actual_won = (cohort == 'won')
            if predicted_won and actual_won:
                t_tp += 1
            elif predicted_won and not actual_won:
                t_fp += 1
            elif not predicted_won and actual_won:
                t_fn += 1
            else:
                t_tn += 1

        n = len(test_opps_list)
        acc = (t_tp + t_tn) / n if n else 0
        prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        mc_data['accuracy'].append(acc)
        mc_data['precision'].append(prec)
        mc_data['recall'].append(rec)
        mc_data['f1'].append(f1)

    mc_results = {k: round(mean(v), 3) for k, v in mc_data.items()}
    print(f"  Accuracy:  {mc_results['accuracy']}")
    print(f"  Precision: {mc_results['precision']}")
    print(f"  Recall:    {mc_results['recall']}")
    print(f"  F1:        {mc_results['f1']}")

    # Deterministic test set metrics
    det_tp = det_fp = det_fn = det_tn = 0
    for opp_name in test_opps_list:
        cohort = all_opps[opp_name]['cohort']
        score = sum(signal_weights[sig['name']] for sig in all_model_signals
                    if signal_results[opp_name][sig['name']])
        predicted_won = score > 0
        actual_won = (cohort == 'won')
        if predicted_won and actual_won:
            det_tp += 1
        elif predicted_won and not actual_won:
            det_fp += 1
        elif not predicted_won and actual_won:
            det_fn += 1
        else:
            det_tn += 1

    n = len(test_opps_list)
    det_acc = (det_tp + det_tn) / n if n else 0
    det_prec = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0
    det_rec = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0
    det_f1 = 2 * det_prec * det_rec / (det_prec + det_rec) if (det_prec + det_rec) > 0 else 0

    det_results = {
        'accuracy': round(det_acc, 3), 'precision': round(det_prec, 3),
        'recall': round(det_rec, 3), 'f1': round(det_f1, 3),
        'tp': det_tp, 'fp': det_fp, 'fn': det_fn, 'tn': det_tn,
    }
    print(f"\n  Deterministic: Acc={det_acc:.3f} P={det_prec:.3f} R={det_rec:.3f} F1={det_f1:.3f}")

    # Extract citations
    print("\n=== EXTRACTING CITATIONS ===")
    citations = {}
    for sig in all_model_signals:
        name = sig['name']
        direction = sig['direction']
        correct_citations = []
        failure_citations = []

        for opp_name in sorted(all_opps.keys()):
            opp_data = all_opps[opp_name]
            fires = signal_results[opp_name][name]
            if not fires:
                continue

            # Get citation: from evaluations if available, else regex
            if use_evaluations:
                snippet = eval_lookup.get(opp_name, {}).get(name, {}).get('citation')
            else:
                snippet = find_citation(opp_data['full_text'], sig)
            if not snippet:
                continue

            cohort = opp_data['cohort']
            earliest_date = min(opp_data['meeting_dates']) if opp_data['meeting_dates'] else 'unknown'

            citation = {
                'opp_name': opp_name,
                'meeting_date': earliest_date,
                'snippet': snippet,
                'cohort': cohort,
            }

            if (direction == 'positive' and cohort == 'won') or \
               (direction == 'negative' and cohort == 'lost'):
                if len(correct_citations) < 5:
                    correct_citations.append(citation)
            else:
                if len(failure_citations) < 3:
                    citation['why_misleading'] = (
                        f"Signal fired but deal was Closed {'Lost' if cohort == 'lost' else 'Won'}. "
                        f"Since opportunities may share underlying meeting recordings, this pattern "
                        f"can appear in both winning and losing deals."
                    )
                    failure_citations.append(citation)

            if len(correct_citations) >= 5 and len(failure_citations) >= 3:
                break

        citations[name] = {'correct': correct_citations, 'failures': failure_citations}
        print(f"  {name}: {len(correct_citations)} correct, {len(failure_citations)} failure")

    # Build final output
    final_model = {
        'signals': [],
        'model_performance': {
            'monte_carlo': mc_results,
            'deterministic_test': det_results,
        },
        'data_characteristics': {
            'total_opportunities': len(all_opps),
            'won_count': len(won_opps),
            'lost_count': len(lost_opps),
            'train_won': len(won_train),
            'train_lost': len(lost_train),
            'test_won': len(won_test),
            'test_lost': len(lost_test),
            'variable_signals': len(tier1),
            'universal_signals': len(tier2),
        },
        'opportunity_lists': {
            'won': sorted(won_opps.keys()),
            'lost': sorted(lost_opps.keys()),
        },
        'candidate_signal_count': len(signals),
        'qualified_signal_count': len(all_model_signals),
    }

    for sig in all_model_signals:
        name = sig['name']
        prev = signal_prevalence[name]
        metrics = signal_metrics.get(name, {})
        weight = signal_weights.get(name, 0)

        sig_entry = {
            'name': name,
            'description': sig['description'],
            'direction': sig['direction'],
            'weight': round(weight, 4),
            'tier': 'variable' if sig in tier1 else 'universal',
            'prevalence': {
                'overall_pct': prev['total_pct'],
                'won_pct': prev['won_pct'],
                'lost_pct': prev['lost_pct'],
                'delta': prev['delta'],
            },
            'test_metrics': metrics,
            'citations': citations.get(name, {'correct': [], 'failures': []}),
        }
        # Include format-specific fields when present
        if 'category' in sig:
            sig_entry['category'] = sig['category']
        if 'patterns' in sig:
            sig_entry['patterns'] = sig['patterns']
        if 'source_meeting' in sig:
            sig_entry['source_meeting'] = sig['source_meeting']

        final_model['signals'].append(sig_entry)

    model_path = f"{output_dir}/final_model.json"
    with open(model_path, 'w') as f:
        json.dump(final_model, f, indent=2)
    print(f"\nFinal model written to {model_path}")

    citations_path = f"{output_dir}/citations.json"
    with open(citations_path, 'w') as f:
        json.dump(citations, f, indent=2)
    print(f"Citations written to {citations_path}")

    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total signals: {len(all_model_signals)} ({len(tier1)} variable, {len(tier2)} universal)")
    print(f"Deterministic test: Acc={det_acc:.3f} P={det_prec:.3f} R={det_rec:.3f} F1={det_f1:.3f}")
    print(f"Monte Carlo: Acc={mc_results['accuracy']} P={mc_results['precision']} "
          f"R={mc_results['recall']} F1={mc_results['f1']}")

    return {
        'deterministic': det_results,
        'monte_carlo': mc_results,
        'signal_count': len(all_model_signals),
    }


def main():
    parser = argparse.ArgumentParser(description='Build predictive signal model from transcript corpus')
    parser.add_argument('--output-dir', required=True, help='Directory containing transcript_corpus.json')
    parser.add_argument('--signals-file', required=True, help='Path to signals JSON file')
    parser.add_argument('--evaluations-file', default=None, help='Path to pre-computed signal evaluations JSON (from generate_signals.py)')
    args = parser.parse_args()

    build_signal_model(args.output_dir, args.signals_file, evaluations_file=args.evaluations_file)


if __name__ == '__main__':
    main()
