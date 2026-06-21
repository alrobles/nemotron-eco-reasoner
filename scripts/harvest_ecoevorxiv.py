#!/usr/bin/env python3
"""
harvest_ecoevorxiv.py — Harvest preprints from ecoevorxiv.org (Janeway API).
Correct API: https://ecoevorxiv.org/api/preprints/?search=TERM&limit=N

ecoevorxiv has 3,370+ ecology preprints. Free, no auth.

Usage:
    python3 harvest_ecoevorxiv.py --limit 100 --output ecoevorxiv_papers.jsonl
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.parse

API = "https://ecoevorxiv.org/api/preprints/"

TOPICS = [
    "species distribution model",
    "climate change biodiversity",
    "conservation biology",
    "macroecology",
    "community ecology",
    "population dynamics",
    "phylogenetic comparative",
    "ecological niche model",
    "biodiversity conservation",
    "ecosystem services",
    "functional trait diversity",
    "invasive species",
    "movement ecology dispersal",
    "disease ecology zoonotic",
    "urban ecology biodiversity",
    "tropical forest biodiversity",
    "marine conservation",
    "freshwater biodiversity",
    "pollinator network",
    "forest ecology management",
    "spatial ecology landscape",
    "occupancy detection model",
    "species distribution abundance",
    "remote sensing land cover",
    "genetic diversity population",
    "phenological climate shift",
    "range shift distribution",
    "extinction risk assessment",
    "restoration ecology",
    "biodiversity monitoring",
    "biogeography species richness",
    "food web trophic network",
    "coexistence competition niche",
    "carbon sequestration ecosystem",
    "meta-analysis ecology",
]


def search_ecoevorxiv(query: str, limit: int = 50, offset: int = 0) -> list[dict]:
    """Search ecoevorxiv via Janeway API."""
    url = f"{API}?search={urllib.parse.quote(query)}&limit={min(limit, 100)}&offset={offset}"
    
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "EcoSeek/1.0 (ecoseek.org)",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return []
    
    results = []
    for item in data.get("results", []):
        abstract = (item.get("abstract") or "").strip()
        title = (item.get("title") or "").strip()
        if not title or not abstract or len(abstract) < 100:
            continue
        
        authors = item.get("authors", [])
        author_names = [f"{a.get('first_name','')} {a.get('last_name','')}".strip() for a in authors]
        subjects = [s.get("name", "") for s in item.get("subject", [])]
        
        results.append({
            "pmid": f"ecoevorxiv:{item.get('pk', '?')}",
            "title": title,
            "abstract": abstract,
            "authors": "; ".join(author_names[:6]),
            "journal": f"ecoevorxiv ({', '.join(subjects[:3])})",
            "pub_year": (item.get("date_published") or "2025")[:4],
            "keywords": "",
            "mesh_terms": "ecoevorxiv:" + (subjects[0] if subjects else ""),
            "doi": item.get("preprint_doi", ""),
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Max papers per topic")
    parser.add_argument("--output", default="ecoevorxiv_papers.jsonl")
    args = parser.parse_args()
    
    print(f"ECOEVORXIV HARVEST: {len(TOPICS)} topics × ~{args.limit} papers")
    print(f"API: {API}")
    print()
    
    seen = set()
    all_papers = []
    topic_counts = {}
    
    for i, topic in enumerate(TOPICS):
        print(f"  [{i+1}/{len(TOPICS)}] '{topic[:50]}...'", end=" ", flush=True)
        try:
            papers = search_ecoevorxiv(topic, limit=args.limit)
            new = [p for p in papers if p["pmid"] not in seen]
            for p in new:
                seen.add(p["pmid"])
                p["topic"] = topic
            topic_counts[topic] = len(new)
            all_papers.extend(new)
            print(f"→ {len(new)} new ({len(papers)} total)")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)
    
    with open(args.output, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*50}")
    print(f"ECOEVORXIV COMPLETE")
    print(f"  Total:  {len(all_papers)} unique papers (from {len(seen)} deduped)")
    print(f"  Output: {args.output}")
    print(f"  Top topics by yield:")
    for label, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(40, count // 3)
        print(f"    {label[:38]:38s} {count:3d} {bar}")


if __name__ == "__main__":
    main()
