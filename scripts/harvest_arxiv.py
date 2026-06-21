#!/usr/bin/env python3
"""
harvest_arxiv.py — Query arXiv API for ecological papers across multiple categories.
arXiv API is free, no auth required. Returns Atom XML, we parse to JSONL.

Categories: q-bio.PE (populations/evolution), q-bio.QM (quantitative methods),
            stat.AP (applications), cs.LG (machine learning)

Usage:
    python3 harvest_arxiv.py --limit 100 --workers 5 --output arxiv_papers.jsonl
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

ARXIV_API = "http://export.arxiv.org/api/query"

# Ecological search queries targeting specific arXiv categories
QUERIES = [
    # Species distribution modeling
    ("species distribution model MaxEnt niche", "sdm", "q-bio.PE"),
    ("ecological niche model habitat suitability", "niche_model", "q-bio.PE"),
    ("biodiversity climate change range shift", "climate", "q-bio.PE"),
    ("species richness diversity gradient macroecology", "macroeco", "q-bio.PE"),
    ("phylogenetic comparative method trait evolution", "phylo", "q-bio.PE"),
    ("community ecology coexistence competition", "community", "q-bio.PE"),
    ("population dynamics viability matrix model", "pop_dynamics", "q-bio.PE"),
    ("food web network ecological interaction", "foodweb", "q-bio.PE"),
    ("conservation planning prioritization biodiversity", "conservation", "q-bio.PE"),
    ("invasive species spread impact management", "invasion", "q-bio.PE"),
    ("disease ecology epidemic model wildlife", "disease", "q-bio.PE"),
    ("movement ecology dispersal migration animal", "movement", "q-bio.PE"),
    ("spatial ecology point pattern landscape", "spatial", "stat.AP"),
    ("Bayesian hierarchical model ecological occupancy", "bayesian", "stat.AP"),
    ("generalized additive model spline ecological", "gam", "stat.AP"),
    ("machine learning deep learning ecology biodiversity", "ml_eco", "cs.LG"),
    ("remote sensing satellite biodiversity land cover", "remote_sensing", "cs.LG"),
    ("neural network species identification classification", "nn_eco", "cs.LG"),
    ("citizen science iNaturalist eBird observation", "citizen", "q-bio.PE"),
    ("marine ocean biodiversity conservation fish", "marine", "q-bio.PE"),
    ("forest carbon biomass climate mitigation", "forest", "q-bio.PE"),
    ("urban ecology biodiversity green space", "urban", "q-bio.PE"),
    ("pollinator bee decline network interaction", "pollination", "q-bio.PE"),
    ("ecosystem service valuation natural capital", "ecoservice", "q-bio.PE"),
    ("tropical deforestation biodiversity loss", "tropical", "q-bio.PE"),
    ("alpine arctic biodiversity climate warming", "alpine", "q-bio.PE"),
    ("freshwater lake river biodiversity water quality", "freshwater", "q-bio.PE"),
    ("grassland savanna fire biodiversity grazing", "grassland", "q-bio.PE"),
    ("island biogeography endemic species radiation", "island", "q-bio.PE"),
    ("genetic diversity conservation population structure", "genetics", "q-bio.PE"),
    # Quantitative biology methods
    ("statistical ecology abundance occupancy detection", "stat_eco", "q-bio.QM"),
    ("integrated population model state space mark recapture", "ipm", "q-bio.QM"),
    ("joint species distribution model latent factor", "jsdm", "q-bio.QM"),
    ("causal inference ecology matching regression", "causal", "stat.AP"),
    ("zero inflated model count data ecology", "zero_infl", "stat.AP"),
]


def search_arxiv(query: str, cat: str, limit: int = 100, year_min: int = 2015) -> list[dict]:
    """Query arXiv API and return parsed papers."""
    # Build query URL
    search_q = f"all:{urllib.parse.quote(query)}"
    if cat:
        search_q += f"+AND+cat:{cat}"
    
    url = (f"{ARXIV_API}?search_query={search_q}"
           f"&start=0&max_results={min(limit, 200)}"
           f"&sortBy=relevance&sortOrder=descending")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EcoSeek/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            xml_data = resp.read().decode("utf-8")
    except Exception as e:
        return []
    
    # Parse Atom XML
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    
    results = []
    try:
        root = ET.fromstring(xml_data)
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.find("atom:id", ns)
            title_el = entry.find("atom:title", ns)
            abstract_el = entry.find("atom:summary", ns)
            published = entry.find("atom:published", ns)
            authors = entry.findall("atom:author/atom:name", ns)
            cats = entry.findall("atom:category", ns)
            
            arxiv_id_str = arxiv_id.text.strip() if arxiv_id is not None else ""
            # Extract ID from http://arxiv.org/abs/XXXX.XXXXX
            short_id = arxiv_id_str.split("/")[-1] if arxiv_id_str else ""
            year_str = published.text[:4] if published is not None else "0"
            
            if int(year_str) < year_min:
                continue
            
            title = " ".join(title_el.text.split()) if title_el is not None else ""
            abstract = " ".join(abstract_el.text.split()) if abstract_el is not None else ""
            
            if not title or not abstract or len(abstract) < 100:
                continue
            
            author_names = [a.text for a in authors if a.text]
            category_list = [c.get("term", "") for c in cats]
            
            results.append({
                "pmid": f"arxiv:{short_id}",
                "title": title,
                "abstract": abstract,
                "authors": "; ".join(author_names[:5]),
                "journal": f"arXiv ({', '.join(category_list[:3])})",
                "pub_year": int(year_str),
                "keywords": "",
                "mesh_terms": f"arxiv:{','.join(category_list[:2])}",
            })
    except ET.ParseError:
        pass
    
    return results


def fetch_query(args):
    query, label, cat, limit = args
    try:
        papers = search_arxiv(query, cat, limit=limit)
        return (label, papers, None)
    except Exception as e:
        return (label, [], str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Max papers per query")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent API calls")
    parser.add_argument("--output", default="arxiv_papers.jsonl")
    args = parser.parse_args()
    
    print(f"ARXIV HARVEST: {len(QUERIES)} queries × {args.limit} papers, {args.workers} workers")
    print(f"Categories: q-bio.PE, q-bio.QM, stat.AP, cs.LG")
    print()
    
    tasks = [(q, l, c, args.limit) for q, l, c in QUERIES]
    topic_results = {}
    errors = []
    t_start = time.time()
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_query, t): t[1] for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            label = futures[fut]
            try:
                lbl, papers, err = fut.result()
                if err:
                    errors.append((lbl, err))
                    print(f"  [{i+1}/{len(QUERIES)}] {lbl:20s} ERROR: {err[:60]}")
                else:
                    topic_results[lbl] = papers
                    print(f"  [{i+1}/{len(QUERIES)}] {lbl:20s} → {len(papers):4d} papers")
            except Exception as e:
                errors.append((label, str(e)))
        # Rate limiting: arXiv asks for polite delays
        time.sleep(0.5)
    
    elapsed = time.time() - t_start
    
    # Deduplicate by arXiv ID
    seen = set()
    all_papers = []
    topic_final = {}
    for label, papers in topic_results.items():
        new = []
        for p in papers:
            if p["pmid"] not in seen:
                seen.add(p["pmid"])
                p["topic"] = label
                new.append(p)
        topic_final[label] = len(new)
        all_papers.extend(new)
    
    with open(args.output, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"ARXIV HARVEST COMPLETE in {elapsed:.1f}s")
    print(f"  Total:  {len(all_papers)} unique papers")
    print(f"  Output: {args.output}")
    if errors:
        print(f"  Errors: {len(errors)}")
    print(f"  Rate:   {len(all_papers)/max(elapsed,1)*60:.0f} papers/min")
    print(f"\n  Top 10 topics:")
    for label, count in sorted(topic_final.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(50, count // 5)
        print(f"    {label:20s} {count:4d} {bar}")


if __name__ == "__main__":
    main()
