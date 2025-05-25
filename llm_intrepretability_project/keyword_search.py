#!/usr/bin/env python3
"""
Standalone script to search for keywords in the Wikitext dataset.
Usage: python keyword_search.py --keyword "Golden Gate" --max_results 20
"""

import argparse
import re
from datasets import load_dataset
from typing import List, Tuple
import json


def search_keyword_in_wikitext(
    keyword: str, 
    max_results: int = 20, 
    context_window: int = 100,
    case_sensitive: bool = False,
    exact_match: bool = False,
    dataset_name: str = "wikitext-103",
    start_index: int = 0,
    scan_size: int = None
) -> Tuple[List[dict], dict]:
    """
    Search for a keyword in the Wikitext dataset and return contexts with position mapping.
    
    Args:
        keyword: The keyword to search for
        max_results: Maximum number of results to return
        context_window: Number of characters before and after the match
        case_sensitive: Whether to perform case-sensitive search
        exact_match: Whether to match whole words only
        dataset_name: Which dataset to use
        start_index: Index to start searching from
        scan_size: Number of texts to scan (None for all)
        
    Returns:
        Tuple of (results_list, position_mapping_dict)
    """
    print(f"Loading {dataset_name} dataset...")
    
    # Load different datasets based on name
    if dataset_name == "wikitext-2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif dataset_name == "wikitext-103":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    elif dataset_name == "wikipedia":
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
    else:
        print(f"Unknown dataset {dataset_name}, defaulting to wikitext-103")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    total_texts = len(dataset)
    end_index = min(total_texts, start_index + scan_size) if scan_size else total_texts
    
    print(f"Dataset size: {total_texts:,} texts")
    print(f"Scanning from index {start_index:,} to {end_index:,}")
    print(f"Searching for '{keyword}' (case_sensitive={case_sensitive}, exact_match={exact_match})")
    
    # Prepare search pattern
    if exact_match:
        pattern = r'\b' + re.escape(keyword) + r'\b'
    else:
        pattern = re.escape(keyword)
    
    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern, flags)
    
    results = []
    position_mapping = {
        "keyword": keyword,
        "total_scanned": 0,
        "total_matches": 0,
        "match_positions": [],
        "density_by_chunk": {},
        "scan_range": (start_index, end_index),
        "chunk_size": 1000  # Group indices into chunks for analysis
    }
    
    chunk_size = position_mapping["chunk_size"]
    chunk_matches = {}  # Track matches per chunk
    
    for idx in range(start_index, end_index):
        text = dataset[idx]["text"]
        position_mapping["total_scanned"] += 1
        
        # Skip empty texts
        if not text.strip():
            continue
        
        # Calculate which chunk this index belongs to
        chunk_id = idx // chunk_size
        if chunk_id not in chunk_matches:
            chunk_matches[chunk_id] = 0
            
        # Find all matches in this text
        matches = list(regex.finditer(text))
        
        if matches:
            chunk_matches[chunk_id] += len(matches)
            position_mapping["match_positions"].append({
                "dataset_index": idx,
                "chunk_id": chunk_id,
                "num_matches": len(matches),
                "text_length": len(text)
            })
        
        for match in matches:
            position_mapping["total_matches"] += 1
            start_pos = match.start()
            end_pos = match.end()
            
            # Extract context around the match
            context_start = max(0, start_pos - context_window)
            context_end = min(len(text), end_pos + context_window)
            context = text[context_start:context_end]
            
            # Calculate relative position of the match within the context
            match_start_in_context = start_pos - context_start
            match_end_in_context = end_pos - context_start
            
            result = {
                "dataset_index": idx,
                "chunk_id": chunk_id,
                "dataset_position_percent": (idx / total_texts) * 100,
                "keyword": keyword,
                "matched_text": text[start_pos:end_pos],
                "context": context,
                "match_start_in_context": match_start_in_context,
                "match_end_in_context": match_end_in_context,
                "match_position_in_text": start_pos,
                "text_length": len(text)
            }
            
            results.append(result)
            
            # Stop if we've reached the maximum number of results
            if len(results) >= max_results:
                break
        
        # Break outer loop too if max results reached
        if len(results) >= max_results:
            break
        
        # Progress indicator
        if (idx - start_index + 1) % 1000 == 0:
            scanned_so_far = idx - start_index + 1
            total_to_scan = end_index - start_index
            progress_pct = (scanned_so_far / total_to_scan) * 100
            print(f"Progress: {scanned_so_far:,}/{total_to_scan:,} ({progress_pct:.1f}%) - Found {len(results)} matches so far...")
    
    # Calculate density by chunk
    position_mapping["density_by_chunk"] = chunk_matches
    
    # Find top chunks with most matches
    top_chunks = sorted(chunk_matches.items(), key=lambda x: x[1], reverse=True)[:10]
    position_mapping["top_chunks"] = [
        {
            "chunk_id": chunk_id,
            "start_index": chunk_id * chunk_size,
            "end_index": (chunk_id + 1) * chunk_size,
            "matches": matches,
            "density": matches / chunk_size
        }
        for chunk_id, matches in top_chunks if matches > 0
    ]
    
    print(f"\nScan complete!")
    print(f"Scanned {position_mapping['total_scanned']:,} texts")
    print(f"Found {position_mapping['total_matches']} total matches in {len(results)} result entries")
    
    return results, position_mapping


def display_results(results: List[dict], highlight_matches: bool = True, show_positions: bool = True):
    """Display search results in a readable format with position information."""
    
    if not results:
        print("No matches found.")
        return
    
    print(f"\n=== SEARCH RESULTS ===")
    print(f"Found {len(results)} matches:\n")
    
    for i, result in enumerate(results, 1):
        print(f"--- Match {i} ---")
        
        if show_positions:
            print(f"📍 Dataset Index: {result['dataset_index']:,}")
            print(f"📊 Position: {result['dataset_position_percent']:.1f}% through dataset")
            print(f"📦 Chunk ID: {result['chunk_id']}")
        else:
            print(f"Dataset Index: {result['dataset_index']}")
            
        print(f"Position in text: {result['match_position_in_text']}")
        print(f"Matched text: '{result['matched_text']}'")
        
        if highlight_matches:
            # Create highlighted context
            context = result['context']
            start = result['match_start_in_context']
            end = result['match_end_in_context']
            
            highlighted = (
                context[:start] + 
                f">>>{context[start:end]}<<<" + 
                context[end:]
            )
            print(f"Context: {highlighted}")
        else:
            print(f"Context: {result['context']}")
        
        print()


def display_position_mapping(position_mapping: dict):
    """Display analysis of where matches are concentrated in the dataset."""
    
    print("\n=== POSITION ANALYSIS ===")
    print(f"🔍 Keyword: '{position_mapping['keyword']}'")
    print(f"📈 Total matches: {position_mapping['total_matches']}")
    print(f"📚 Texts scanned: {position_mapping['total_scanned']:,}")
    print(f"🎯 Match rate: {position_mapping['total_matches'] / position_mapping['total_scanned'] * 100:.3f}% of texts")
    
    scan_start, scan_end = position_mapping['scan_range']
    print(f"📊 Scan range: {scan_start:,} to {scan_end:,}")
    
    if position_mapping['top_chunks']:
        print(f"\n🏆 TOP CHUNKS (highest match density):")
        print("Chunk ID | Start Index | End Index   | Matches | Density")
        print("-" * 55)
        
        for chunk_info in position_mapping['top_chunks'][:5]:
            chunk_id = chunk_info['chunk_id']
            start_idx = chunk_info['start_index']
            end_idx = chunk_info['end_index']
            matches = chunk_info['matches']
            density = chunk_info['density']
            
            print(f"{chunk_id:8} | {start_idx:10,} | {end_idx:10,} | {matches:7} | {density:.4f}")
        
        print(f"\n💡 To focus your analysis on high-density areas, use:")
        best_chunk = position_mapping['top_chunks'][0]
        start_suggestion = best_chunk['start_index']
        print(f"   --start_index {start_suggestion} --scan_size 5000")
    
    if len(position_mapping['match_positions']) > 1:
        positions = [m['dataset_index'] for m in position_mapping['match_positions']]
        print(f"\n📍 Match distribution:")
        print(f"   First match at index: {min(positions):,}")
        print(f"   Last match at index: {max(positions):,}")
        print(f"   Spread across: {max(positions) - min(positions):,} indices")


def save_position_mapping(position_mapping: dict, output_file: str):
    """Save the position mapping for later use in streaming."""
    
    # Add some useful derived information
    if position_mapping['top_chunks']:
        # Create streaming recommendations
        streaming_suggestions = []
        for chunk in position_mapping['top_chunks'][:3]:
            streaming_suggestions.append({
                "description": f"High density chunk {chunk['chunk_id']}",
                "start_index": chunk['start_index'],
                "recommended_scan_size": min(10000, chunk['end_index'] - chunk['start_index']),
                "expected_matches": chunk['matches'],
                "density": chunk['density']
            })
        
        position_mapping['streaming_suggestions'] = streaming_suggestions
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(position_mapping, f, indent=2, ensure_ascii=False)
    print(f"📁 Position mapping saved to {output_file}")
    print(f"💡 Use this file to optimize your LLM analysis streaming!")


def save_results(results: List[dict], output_file: str):
    """Save results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Search for keywords in datasets with position mapping for streaming optimization"
    )
    parser.add_argument(
        "--keyword", 
        type=str, 
        required=True,
        help="Keyword to search for (e.g., 'Golden Gate')"
    )
    parser.add_argument(
        "--max_results", 
        type=int, 
        default=20,
        help="Maximum number of results to return (default: 20)"
    )
    parser.add_argument(
        "--context_window", 
        type=int, 
        default=100,
        help="Number of characters before and after match (default: 100)"
    )
    parser.add_argument(
        "--case_sensitive", 
        action="store_true",
        help="Perform case-sensitive search"
    )
    parser.add_argument(
        "--exact_match", 
        action="store_true",
        help="Match whole words only"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--no_highlight", 
        action="store_true",
        help="Don't highlight matches in display"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="wikitext-103",
        choices=["wikitext-2", "wikitext-103", "wikipedia"],
        help="Which dataset to search (default: wikitext-103)"
    )
    parser.add_argument(
        "--start_index", 
        type=int, 
        default=0,
        help="Dataset index to start searching from (default: 0)"
    )
    parser.add_argument(
        "--scan_size", 
        type=int,
        help="Number of texts to scan from start_index (default: scan all)"
    )
    parser.add_argument(
        "--save_mapping", 
        type=str,
        help="Save position mapping to file for streaming optimization"
    )
    parser.add_argument(
        "--show_analysis", 
        action="store_true",
        help="Show detailed position analysis"
    )
    
    args = parser.parse_args()
    
    # Perform search with position mapping
    results, position_mapping = search_keyword_in_wikitext(
        keyword=args.keyword,
        max_results=args.max_results,
        context_window=args.context_window,
        case_sensitive=args.case_sensitive,
        exact_match=args.exact_match,
        dataset_name=args.dataset,
        start_index=args.start_index,
        scan_size=args.scan_size
    )
    
    # Display results
    display_results(results, highlight_matches=not args.no_highlight)
    
    # Show position analysis if requested or if no specific output requested
    if args.show_analysis or (not args.output and not args.save_mapping):
        display_position_mapping(position_mapping)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)
    
    # Save position mapping if requested
    if args.save_mapping:
        save_position_mapping(position_mapping, args.save_mapping)
    
    # Always save a basic mapping file for convenience
    if not args.save_mapping:
        mapping_filename = f"{args.keyword.replace(' ', '_')}_mapping.json"
        save_position_mapping(position_mapping, mapping_filename)


if __name__ == "__main__":
    main()