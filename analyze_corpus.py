#!/usr/bin/env python3
"""
Consolidate all downloaded transcripts into a corpus and analyze unique meetings.

Library entry point:
    consolidate_corpus(output_dir, final_stages, exclude_prefix="[test")

Standalone:
    python3 analyze_corpus.py --output-dir ./output --final-stages "Closed Won,Closed Lost" [--exclude-prefix "[test"]

Outputs (in output_dir):
    transcript_corpus.json  — full corpus with per-opportunity text, metadata
    meeting_analysis.json   — per-unique-meeting statistics (no full text)
    meeting_texts.json      — full text of each unique meeting (human reads this)
"""

import json
import os
import re
import sys
import hashlib
import argparse
from collections import defaultdict, Counter

FILENAME_RE = re.compile(r'^(.+?)_(\d{4}-\d{2}-\d{2})(?:_\d+)?\.json$')


def file_content_hash(filepath):
    """Compute SHA256 hash of file content (first 12 hex chars)."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def process_transcript(filepath):
    """Read a single transcript JSON, return filtered sentences and metadata."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    sentences = data.get('transcript', {}).get('sentences', [])
    filtered = [s for s in sentences if s.get('speaker', '').lower() != 'reevo notetaker']

    text_parts = [s['text'] for s in filtered]
    speakers = set(s['speaker'] for s in filtered)

    return {
        'text': ' '.join(text_parts),
        'sentences': filtered,
        'sentence_count': len(filtered),
        'word_count': sum(len(s['text'].split()) for s in filtered),
        'speakers': list(speakers),
    }


def process_folder(folder_path, exclude_prefix):
    """Process all transcripts in a folder, grouped by opportunity."""
    opps = defaultdict(lambda: {'files': []})

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.json'):
            continue
        if exclude_prefix and fname.startswith(exclude_prefix):
            continue

        m = FILENAME_RE.match(fname)
        if not m:
            print(f"  WARNING: Could not parse filename: {fname}")
            continue

        opp_name = m.group(1)
        meeting_date = m.group(2)
        filepath = os.path.join(folder_path, fname)
        content_hash = file_content_hash(filepath)

        opps[opp_name]['files'].append({
            'filepath': filepath,
            'filename': fname,
            'meeting_date': meeting_date,
            'content_hash': content_hash,
        })

    results = {}
    for opp_name, opp_data in sorted(opps.items()):
        all_text_parts = []
        all_speakers = set()
        total_sentences = 0
        total_words = 0
        meeting_dates = []
        meeting_hashes = []

        for file_info in sorted(opp_data['files'], key=lambda x: x['meeting_date']):
            transcript = process_transcript(file_info['filepath'])
            all_text_parts.append(transcript['text'])
            all_speakers.update(transcript['speakers'])
            total_sentences += transcript['sentence_count']
            total_words += transcript['word_count']
            meeting_dates.append(file_info['meeting_date'])
            meeting_hashes.append(file_info['content_hash'])

        results[opp_name] = {
            'full_text': ' '.join(all_text_parts),
            'meeting_count': len(opp_data['files']),
            'meeting_dates': meeting_dates,
            'meeting_hashes': sorted(set(meeting_hashes)),
            'speakers': sorted(all_speakers),
            'sentence_count': total_sentences,
            'word_count': total_words,
        }

    return results


def analyze_unique_meetings(won_opps, lost_opps, won_dir, lost_dir, exclude_prefix):
    """Analyze each unique meeting recording once."""
    hash_to_file = {}
    for folder in [won_dir, lost_dir]:
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.json'):
                continue
            if exclude_prefix and fname.startswith(exclude_prefix):
                continue
            filepath = os.path.join(folder, fname)
            h = file_content_hash(filepath)
            if h not in hash_to_file:
                hash_to_file[h] = filepath

    hash_presence = defaultdict(lambda: {'won': 0, 'lost': 0, 'total': 0})
    for opp_name, opp_data in won_opps.items():
        for h in set(opp_data['meeting_hashes']):
            hash_presence[h]['won'] += 1
            hash_presence[h]['total'] += 1
    for opp_name, opp_data in lost_opps.items():
        for h in set(opp_data['meeting_hashes']):
            hash_presence[h]['lost'] += 1
            hash_presence[h]['total'] += 1

    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'is', 'it', 'that', 'this', 'was', 'are', 'be', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'we',
        'they', 'he', 'she', 'my', 'your', 'our', 'their', 'me', 'him', 'her', 'us', 'them',
        'so', 'if', 'then', 'than', 'when', 'where', 'how', 'what', 'which', 'who', 'whom',
        'not', 'no', 'yes', 'yeah', 'just', 'like', 'um', 'uh', 'right', 'okay', 'oh', 'well',
        'know', 'think', 'going', 'gonna', 'got', 'get', 'one', 'also', 'really', 'very',
        'about', 'been', 'there', 'here', 'some', 'all', 'any', 'more', 'other', 'its',
        "it's", "don't", "i'm", "we're", "that's", "there's", "they're", "you're", "i've",
        "we've", "what's", "let's", "didn't", "doesn't", "wasn't", "weren't", "won't",
        "can't", "couldn't", "wouldn't", "shouldn't", "haven't", "hasn't", "isn't",
        'from', 'into', 'over', 'after', 'before', 'between', 'through', 'during', 'up',
        'down', 'out', 'off', 'way', 'kind', 'thing', 'things', 'lot', 'actually', 'basically',
        'need', 'want', 'make', 'say', 'said', 'see', 'look', 'come', 'go', 'take', 'give',
        'being', 'those', 'these', 'even', 'still', 'back', 'much', 'many', 'own', 'same',
        'able', 'because', 'since', 'while', 'both', 'each', 'every', 'such', 'only', 'most',
        'new', 'good', 'great', 'sure', 'part', 'point', 'time', 'now', 'then', 'already',
    }

    meetings = {}
    meetings_text = {}

    for h, filepath in sorted(hash_to_file.items()):
        with open(filepath, 'r') as f:
            data = json.load(f)

        sentences = data.get('transcript', {}).get('sentences', [])
        filtered = [s for s in sentences if s.get('speaker', '').lower() != 'reevo notetaker']

        speaker_stats = defaultdict(lambda: {'sentences': 0, 'words': 0})
        for s in filtered:
            speaker_stats[s['speaker']]['sentences'] += 1
            speaker_stats[s['speaker']]['words'] += len(s['text'].split())

        total_words = sum(ss['words'] for ss in speaker_stats.values())
        total_sents = len(filtered)

        questions = sum(1 for s in filtered if '?' in s['text'])
        question_rate = questions / total_sents if total_sents > 0 else 0

        if filtered:
            start_ts = filtered[0].get('start_timestamp', 0)
            end_ts = filtered[-1].get('end_timestamp', 0)
            duration_mins = (end_ts - start_ts) / 60000.0
        else:
            duration_mins = 0

        if speaker_stats:
            max_speaker_words = max(ss['words'] for ss in speaker_stats.values())
            dominance = max_speaker_words / total_words if total_words > 0 else 0
        else:
            dominance = 0

        all_words_lower = []
        for s in filtered:
            words = re.findall(r'\b[a-z]+\b', s['text'].lower())
            all_words_lower.extend([w for w in words if w not in stopwords and len(w) > 2])

        bigrams = Counter()
        trigrams = Counter()
        for i in range(len(all_words_lower) - 1):
            bigrams[(all_words_lower[i], all_words_lower[i + 1])] += 1
        for i in range(len(all_words_lower) - 2):
            trigrams[(all_words_lower[i], all_words_lower[i + 1], all_words_lower[i + 2])] += 1

        full_text = ' '.join(s['text'] for s in filtered)
        meetings_text[h] = full_text

        meetings[h] = {
            'filepath': filepath,
            'sentence_count': total_sents,
            'word_count': total_words,
            'speakers': {spk: dict(stats) for spk, stats in speaker_stats.items()},
            'question_rate': round(question_rate, 3),
            'duration_mins': round(duration_mins, 1),
            'speaker_dominance': round(dominance, 3),
            'avg_turn_length_words': round(total_words / total_sents, 1) if total_sents > 0 else 0,
            'top_bigrams': [(' '.join(bg), cnt) for bg, cnt in bigrams.most_common(20)],
            'top_trigrams': [(' '.join(tg), cnt) for tg, cnt in trigrams.most_common(20)],
            'presence': hash_presence[h],
        }

    return meetings, meetings_text


def _resolve_won_lost_dirs(output_dir, final_stages):
    """Map stage names containing 'won'/'lost' to transcript directories.

    Uses the '{output_dir}/{StageName} Transcripts/' convention established
    by download_transcripts.py.
    """
    won_dir = None
    lost_dir = None
    for stage in final_stages:
        stage_lower = stage.lower()
        dir_path = os.path.join(output_dir, f"{stage} Transcripts")
        if 'won' in stage_lower:
            won_dir = dir_path
        elif 'lost' in stage_lower:
            lost_dir = dir_path
    return won_dir, lost_dir


def consolidate_corpus(output_dir, final_stages, exclude_prefix="[test"):
    """Library entry point: consolidate transcripts and write analysis files.

    Args:
        output_dir: Root output directory containing transcript folders.
        final_stages: List of stage name strings (e.g. ["Closed Won", "Closed Lost"]).
        exclude_prefix: Filename prefix to skip during analysis.

    Returns:
        dict with keys: won_count, lost_count, unique_meetings
    """
    won_dir, lost_dir = _resolve_won_lost_dirs(output_dir, final_stages)

    if not won_dir or not os.path.isdir(won_dir):
        print(f"ERROR: Won directory not found: {won_dir}")
        sys.exit(1)
    if not lost_dir or not os.path.isdir(lost_dir):
        print(f"ERROR: Lost directory not found: {lost_dir}")
        sys.exit(1)

    print("Processing Won transcripts...")
    won = process_folder(won_dir, exclude_prefix)
    print(f"  Found {len(won)} Won opportunities")

    print("Processing Lost transcripts...")
    lost = process_folder(lost_dir, exclude_prefix)
    print(f"  Found {len(lost)} Lost opportunities")

    # Corpus output
    corpus = {'won': won, 'lost': lost}
    corpus_path = os.path.join(output_dir, 'transcript_corpus.json')
    with open(corpus_path, 'w') as f:
        json.dump(corpus, f, indent=2)
    print(f"Corpus written to {corpus_path}")

    # Meeting analysis
    print("\nAnalyzing unique meetings...")
    meetings, meetings_text = analyze_unique_meetings(won, lost, won_dir, lost_dir, exclude_prefix)

    analysis_path = os.path.join(output_dir, 'meeting_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(meetings, f, indent=2)

    texts_path = os.path.join(output_dir, 'meeting_texts.json')
    with open(texts_path, 'w') as f:
        json.dump(meetings_text, f, indent=2)

    print(f"  Found {len(meetings)} unique meetings")

    # Summary stats
    print("\n=== CORPUS SUMMARY ===")
    print(f"Won opportunities: {len(won)}")
    print(f"Lost opportunities: {len(lost)}")
    print(f"Total opportunities: {len(won) + len(lost)}")

    all_hashes = set()
    for opp in list(won.values()) + list(lost.values()):
        all_hashes.update(opp['meeting_hashes'])
    print(f"Unique meeting hashes: {len(all_hashes)}")

    won_counts = Counter(opp['meeting_count'] for opp in won.values())
    lost_counts = Counter(opp['meeting_count'] for opp in lost.values())
    print(f"Won meeting counts: {dict(won_counts)}")
    print(f"Lost meeting counts: {dict(lost_counts)}")

    print("\n=== MEETING PRESENCE ===")
    total_opps = len(won) + len(lost)
    for h, m in sorted(meetings.items()):
        p = m['presence']
        print(f"  {h}: {p['total']}/{total_opps} opps ({p['won']} won, {p['lost']} lost), "
              f"{m['sentence_count']} sentences, speakers: {list(m['speakers'].keys())}")

    # Find opps with fewer meetings than the mode
    all_opps_combined = {**won, **lost}
    if all_opps_combined:
        mode_count = Counter(opp['meeting_count'] for opp in all_opps_combined.values()).most_common(1)[0][0]
        print(f"\n=== OPPS WITH < {mode_count} MEETINGS ===")
        for opp_name, opp in sorted(all_opps_combined.items()):
            if opp['meeting_count'] < mode_count:
                cohort = 'won' if opp_name in won else 'lost'
                missing = all_hashes - set(opp['meeting_hashes'])
                print(f"  {opp_name} ({cohort}): {opp['meeting_count']} meetings, missing: {missing}")

    return {
        'won_count': len(won),
        'lost_count': len(lost),
        'unique_meetings': len(meetings),
    }


def main():
    parser = argparse.ArgumentParser(description='Consolidate transcripts into corpus')
    parser.add_argument('--output-dir', required=True, help='Root output directory')
    parser.add_argument('--final-stages', required=True, help='Comma-separated final stages (e.g. "Closed Won,Closed Lost")')
    parser.add_argument('--exclude-prefix', default='[test', help='Filename prefix to exclude')
    args = parser.parse_args()

    final_stages = [s.strip() for s in args.final_stages.split(",")]
    consolidate_corpus(args.output_dir, final_stages, args.exclude_prefix)


if __name__ == '__main__':
    main()
