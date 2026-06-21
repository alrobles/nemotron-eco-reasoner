#!/usr/bin/env python3
"""
harvest_ecoevorxiv.py — Harvest preprints from ecoevorxiv.org (OSF Preprints API).
ecoevorxiv is the premier preprint server for ecology, evolution, and conservation.

API: https://ecoevorxiv.org/api/v2/preprints/  (OSF v2 API)
No auth required for reading public preprints.

Usage:
    python3 harvest_ecoevorxiv.py --limit 100 --output ecoevorxiv_papers.jsonl
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

API = "https://api.osf.io/v2/preprints/"
FILTER = "?filter[provider]=ecoevorxiv"
PAGE_SIZE = 50

# Also try ecoevorxiv direct search endpoint
SEARCH_API = "https://ecoevorxiv.org/api/v1/search/preprints/"

# Ecological keyword searches  
TOPICS = [
    "species distribution model",
    "climate change biodiversity",
    "conservation biology",
    "macroecology",
    "community ecology",
    "population dynamics",
    "phylogenetic comparative",
    "ecological niche",
    "biodiversity hotspot",
    "ecosystem service",
    "trait-based ecology",
    "functional diversity",
    "invasion biology",
    "movement ecology",
    "disease ecology",
    "urban ecology",
    "tropical ecology",
    "marine biodiversity",
    "freshwater ecology",
    "pollination network",
    "forest ecology",
    "grassland ecology",
    "spatial ecology",
    "occupancy model",
    "joint species distribution",
    "remote sensing biodiversity",
    "genetic diversity conservation",
    "phenological change",
    "range shift",
    "extinction risk",
]


def search_ecoevorxiv(query: str, limit: int = 50) -> list[dict]:
    """Search ecoevorxiv via OSF API."""
    url = f"{SEARCH_API}?q={urllib.request.quote(query)}&size={min(limit, 50)}"
    
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "EcoSeek/1.0 (ecoseek.org)",
            "Accept": "application/vnd.api+json",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return []
    
    results = []
    hits = data.get("hits", {}).get("hits", [])
    for hit in hits:
        src = hit.get("_source", {})
        title = src.get("title", "")
        abstract = src.get("abstract", "")
        if not title or not abstract or len(abstract) < 100:
            continue
        
        results.append({
            "pmid": f"ecoevorxiv:{src.get('guid', src.get('id', '?'))}",
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(src.get("authors", [])[:5]),
            "journal": f"ecoevorxiv ({src.get('subject_area', 'ecology')})",
            "pub_year": src.get("publication_year", src.get("date_published", "2024")[:4]),
            "keywords": "",
            "mesh_terms": "ecoevorxiv:" + (src.get("subject_area", "")),
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Max papers per topic")
    parser.add_argument("--output", default="ecoevorxiv_papers.jsonl")
    args = parser.parse_args()
    
    print(f"ECOEVORXIV HARVEST: {len(TOPICS)} topics × {args.limit} papers")
    
    seen = set()
    all_papers = []
    topic_counts = {}
    
    for i, topic in enumerate(TOPICS):
        print(f"  [{i+1}/{len(TOPICS)}] '{topic}'...", end=" ", flush=True)
        try:
            papers = search_ecoevorxiv(topic, limit=args.limit)
            new = [p for p in papers if p["pmid"] not in seen]
            for p in new:
                seen.add(p["pmid"])
                p["topic"] = topic
            topic_counts[topic] = len(new)
            all_papers.extend(new)
            print(f"→ {len(new)} new")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)  # Rate limit
    
    with open(args.output, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*50}")
    print(f"ECOEVORXIV COMPLETE")
    print(f"  Total:  {len(all_papers)} unique papers")
    print(f"  Output: {args.output}")
    print(f"  Top topics:")
    for label, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(40, count // 3)
        print(f"    {label[:40]:40s} {count:3d} {bar}")


if __name__ == "__main__":
    main()
