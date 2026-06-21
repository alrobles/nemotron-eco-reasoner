#!/usr/bin/env python3
"""
harvest_scielo.py — Harvest ecology preprints from SciELO Preprints API.
SciELO is the largest Latin American / Global South open science repository.
Lots of tropical ecology, biodiversity, and conservation papers. Free API.

API: https://preprints.scielo.org/api/v1/preprints/?search=TERM&limit=N

Usage:
    python3 harvest_scielo.py --limit 100 --output scielo_papers.jsonl
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.parse

API = "https://preprints.scielo.org/api/v1/preprints/"

TOPICS = [
    "ecologia",
    "biodiversidad",
    "conservacion",
    "cambio climatico",
    "especies",
    "bosque tropical",
    "deforestacion",
    "restauracion ecologica",
    "areas protegidas",
    "servicios ecosistemicos",
    "ecologia de poblaciones",
    "comunidades vegetales",
    "distribucion de especies",
    "macroecologia",
    "biogeografia",
    "ecologia funcional",
    "polinizacion",
    "invasion biologica",
    "ecologia urbana",
    "ecologia marina",
    "ecologia de agua dulce",
    "conservacion de suelos",
    "agroecologia",
    "etnobiologia",
    "ecologia del paisaje",
    "modelado de nicho ecologico",
    "genetica de la conservacion",
    "ecologia de enfermedades",
    "ecologia del fuego",
    "ecologia isotopica",
]


def search_scielo(query: str, limit: int = 50, offset: int = 0) -> list[dict]:
    """Search SciELO Preprints API."""
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
        abstract = ""
        for lang in ["en", "es", "pt"]:
            a = (item.get(f"abstract_{lang}") or "").strip()
            if a:
                abstract = a
                break
        
        title = ""
        for lang in ["en", "es", "pt"]:
            t = (item.get(f"title_{lang}") or "").strip()
            if t:
                title = t
                break
        
        if not title or not abstract or len(abstract) < 100:
            continue
        
        authors = item.get("authors", [])
        author_names = [a.get("name", "") for a in authors if a.get("name")]
        
        results.append({
            "pmid": f"scielo:{item.get('preprint_pid', item.get('id', '?'))}",
            "title": title,
            "abstract": abstract,
            "authors": "; ".join(author_names[:5]),
            "journal": f"SciELO Preprints ({item.get('language', 'multi')})",
            "pub_year": (item.get("date_submitted") or "2025")[:4],
            "keywords": "",
            "mesh_terms": "scielo:ecologia_latam",
            "doi": item.get("doi", ""),
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Max papers per topic")
    parser.add_argument("--output", default="scielo_papers.jsonl")
    args = parser.parse_args()
    
    print(f"SCIELO HARVEST: {len(TOPICS)} Spanish/Portuguese ecology topics × {args.limit} papers")
    print(f"API: {API}")
    print()
    
    seen = set()
    all_papers = []
    topic_counts = {}
    
    for i, topic in enumerate(TOPICS):
        print(f"  [{i+1}/{len(TOPICS)}] '{topic}'", end=" ", flush=True)
        try:
            papers = search_scielo(topic, limit=args.limit)
            new = [p for p in papers if p["pmid"] not in seen]
            for p in new:
                seen.add(p["pmid"])
                p["topic"] = topic
            topic_counts[topic] = len(new)
            all_papers.extend(new)
            print(f"→ {len(new)} new")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)
    
    with open(args.output, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*50}")
    print(f"SCIELO COMPLETE: {len(all_papers)} papers")
    print(f"  Output: {args.output}")
    for label, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(40, count // 3)
        print(f"    {label[:38]:38s} {count:3d} {bar}")


if __name__ == "__main__":
    main()
