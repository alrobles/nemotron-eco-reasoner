#!/usr/bin/env python3
"""
harvest_biorxiv.py — Harvest ecology preprints from bioRxiv API.
Crawls month-by-month across 2020-2026, filtering by ecological categories.
bioRxiv categories: ecology, evolutionary biology, zoology, plant biology

API: https://api.biorxiv.org/details/biorxiv/YYYY-MM-DD/YYYY-MM-DD/cursor

Usage:
    python3 harvest_biorxiv.py --limit 200 --output biorxiv_papers.jsonl
"""

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta

API = "https://api.biorxiv.org/details/biorxiv/"

# Actual bioRxiv category names (verify from API response)
ECO_CATS = {"ecology", "evolutionary biology", "zoology", "plant biology"}


def fetch_month(start_date: str, end_date: str, cursor: int = 0) -> list[dict]:
    """Fetch one page from bioRxiv API."""
    url = f"{API}{start_date}/{end_date}/{cursor}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return []
    
    results = []
    for paper in data.get("collection", []):
        cat = (paper.get("category") or "").lower()
        if cat not in ECO_CATS:
            continue
        abstract = (paper.get("abstract") or "").strip()
        if len(abstract) < 100:
            continue
        
        results.append({
            "pmid": f"biorxiv:{paper.get('doi', paper.get('paper_id', '?'))}",
            "title": (paper.get("title") or "").strip(),
            "abstract": abstract,
            "authors": (paper.get("authors") or "").strip(),
            "journal": f"bioRxiv ({paper.get('category', 'ecology')})",
            "pub_year": (paper.get("date") or "2025")[:4],
            "keywords": "",
            "mesh_terms": f"biorxiv:{cat}",
            "doi": paper.get("doi", ""),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="Max papers per month")
    parser.add_argument("--output", default="biorxiv_papers.jsonl")
    parser.add_argument("--year-start", type=int, default=2020)
    parser.add_argument("--year-end", type=int, default=2026)
    args = parser.parse_args()
    
    print(f"BIORXIV HARVEST: {args.year_start}-{args.year_end}, max {args.limit}/month")
    print(f"Categories: {ECO_CATS}")
    print()
    
    seen = set()
    all_papers = []
    
    current = datetime(args.year_start, 1, 1)
    end = datetime(args.year_end, 12, 31)
    
    month_count = 0
    while current <= end:
        month_end = (current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        s = current.strftime("%Y-%m-%d")
        e = month_end.strftime("%Y-%m-%d")
        
        collected = 0
        cursor = 0
        batch_new = 0
        
        print(f"  {s} to {e}", end=" ", flush=True)
        
        while cursor < 300:  # max 3 pages per month
            papers = fetch_month(s, e, cursor)
            if not papers:
                break
            for p in papers:
                if p["pmid"] not in seen:
                    seen.add(p["pmid"])
                    all_papers.append(p)
                    collected += 1
            cursor += 30
            time.sleep(0.3)
            if collected >= args.limit:
                break
        
        print(f"→ {collected} new (total: {len(all_papers)})")
        month_count += 1
        
        current = month_end + timedelta(days=1)
        time.sleep(0.5)
    
    with open(args.output, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*50}")
    print(f"BIORXIV COMPLETE: {len(all_papers)} papers from {month_count} months")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
