#!/usr/bin/env python3
"""
harvest_biorxiv.py — Harvest ecology preprints from bioRxiv API.
bioRxiv is the largest biology preprint server. Free, no auth.

API: https://api.biorxiv.org/details/biorxiv/YYYY-MM-DD/YYYY-MM-DD/cursor
Also supports full-text search via content_summary endpoint.

Usage:
    python3 harvest_biorxiv.py --limit 200 --output biorxiv_papers.jsonl
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.parse

# bioRxiv content API — search by category/topic
SEARCH_API = "https://api.biorxiv.org/details/biorxiv/"

# Ecological subject areas in bioRxiv
COLLECTIONS = [
    "ecology",
    "evolutionary-biology",
    "zoology",
    "plant-biology",
    "microbiology",
    "genetics-population",
    "environmental-science",
]


def search_biorxiv(keyword: str, limit: int = 100, cursor: int = 0) -> list[dict]:
    """Search bioRxiv by keyword via the details API with text search."""
    # bioRxiv API uses date ranges. For text search, use the content_detail endpoint
    url = f"https://api.biorxiv.org/details/biorxiv/2015-01-01/2026-12-31/{cursor}"
    
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "EcoSeek/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  API error ({cursor}): {e}", file=sys.stderr)
        return []
    
    results = []
    papers = data.get("collection", [])
    
    keyword_lower = keyword.lower()
    for paper in papers:
        title = (paper.get("title") or "").lower()
        abstract = (paper.get("abstract") or "").lower()
        category = (paper.get("category") or "").lower()
        
        # Filter by keyword match and category
        if keyword_lower not in title and keyword_lower not in abstract:
            continue
        if category not in COLLECTIONS:
            continue
        
        if len(paper.get("abstract") or "") < 100:
            continue
        
        results.append({
            "pmid": f"biorxiv:{paper.get('doi', paper.get('paper_id', '?'))}",
            "title": (paper.get("title") or "").strip(),
            "abstract": (paper.get("abstract") or "").strip(),
            "authors": (paper.get("authors") or "").strip(),
            "journal": f"bioRxiv ({paper.get('category', 'ecology')})",
            "pub_year": (paper.get("date") or "2025")[:4],
            "keywords": "",
            "mesh_terms": f"biorxiv:{paper.get('category', '')}",
            "doi": paper.get("doi", ""),
        })
        
        if len(results) >= limit:
            break
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Papers per keyword batch")
    parser.add_argument("--keywords", type=str, default="ecology,conservation,biodiversity,climate,evolution,species,niche,population,community,ecosystem")
    parser.add_argument("--output", default="biorxiv_papers.jsonl")
    args = parser.parse_args()
    
    keywords = [k.strip() for k in args.keywords.split(",")]
    print(f"BIORXIV HARVEST: {len(keywords)} keywords × up to {args.limit} papers each")
    print(f"Categories: {', '.join(COLLECTIONS)}")
    print()
    
    seen = set()
    all_papers = []
    kw_counts = {}
    
    for i, kw in enumerate(keywords):
        batch = 0
        collected = 0
        max_batches = 3  # cursor pages per keyword
        
        print(f"  [{i+1}/{len(keywords)}] '{kw}'", end=" ", flush=True)
        
        for cursor in range(0, max_batches * 100, 100):
            papers = search_biorxiv(kw, limit=args.limit - collected, cursor=cursor)
            new = [p for p in papers if p["pmid"] not in seen]
            for p in new:
                seen.add(p["pmid"])
                p["topic"] = kw
            all_papers.extend(new)
            collected += len(new)
            
            if len(papers) < 100:  # no more results
                break
            if collected >= args.limit:
                break
            time.sleep(0.5)
        
        kw_counts[kw] = collected
        print(f"→ {collected} papers")
        time.sleep(0.5)
    
    with open(args.output, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*50}")
    print(f"BIORXIV COMPLETE: {len(all_papers)} papers")
    print(f"  Output: {args.output}")
    for kw, n in sorted(kw_counts.items(), key=lambda x: -x[1]):
        print(f"    {kw:20s} {n:4d}")


if __name__ == "__main__":
    main()
