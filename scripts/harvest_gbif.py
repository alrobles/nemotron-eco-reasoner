#!/usr/bin/env python3
"""
harvest_gbif.py — Query GBIF Literature FTS5 for ecological papers.
GBIF Literature has 61,561 papers that cite biodiversity data.

Usage:
    python3 harvest_gbif.py --limit 500 --workers 8 --output gbif_papers.jsonl

Requires: GBIF Literature FTS5 at /home/a474r867/work/gbif_literature/gbif_literature_fts.db
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = "/home/a474r867/work/gbif_literature/gbif_literature_fts.db"

# Queries tailored for GBIF literature (papers that use biodiversity data)
TOPICS = [
    ("species distribution model MaxEnt niche", "sdm"),
    ("climate change biodiversity range shift", "climate"),
    ("invasive species impact spread", "invasion"),
    ("conservation protected area biodiversity", "conservation"),
    ("species richness diversity gradient", "macroeco"),
    ("phylogenetic diversity evolution trait", "phylo"),
    ("ecological niche overlap competition", "niche"),
    ("land use land cover change fragmentation", "landuse"),
    ("species occurrence record sampling bias", "occurrence"),
    ("biodiversity informatics data quality", "informatics"),
    ("ecosystem service natural capital", "ecoservice"),
    ("functional diversity trait community", "functional"),
    ("urban ecology biodiversity city", "urban"),
    ("pollination network plant pollinator", "pollination"),
    ("forest biomass carbon remote sensing", "forest"),
    ("marine biodiversity ocean conservation", "marine"),
    ("freshwater biodiversity stream river", "freshwater"),
    ("disease vector host biodiversity dilution", "disease"),
    ("agriculture biodiversity intensification", "agriculture"),
    ("mountain alpine biodiversity elevation", "alpine"),
    ("island biogeography endemic species", "island"),
    ("tropical biodiversity deforestation", "tropical"),
    ("arctic boreal biodiversity climate warming", "arctic"),
    ("restoration rewilding ecosystem recovery", "restoration"),
    ("citizen science monitoring observation", "citizen"),
    ("remote sensing satellite biodiversity", "remote_sensing"),
    ("genetic diversity population structure conservation", "genetics"),
    ("ecological modeling Bayesian hierarchical", "modeling"),
    ("bird mammal amphibian reptile distribution", "vertebrates"),
    ("insect pollinator decline biodiversity", "insects"),
    ("plant diversity vegetation community", "plants"),
    ("fungi mycorrhizal microbial diversity", "microbial"),
    ("soil biodiversity ecosystem function", "soil"),
    ("wetland biodiversity hydrology conservation", "wetland"),
    ("grassland savanna biodiversity fire", "grassland"),
    ("coral reef biodiversity bleaching climate", "coral"),
    ("fisheries marine protected area overfishing", "fisheries"),
    ("endangered threatened species extinction risk", "endangered"),
    ("zoogeography biogeographic region Wallace", "zoogeo"),
    ("biodiversity hotspot endemic richness threat", "hotspot"),
]


def search_gbif(query: str, limit: int = 500, year_min: int = 2015) -> list[dict]:
    """Query GBIF Literature FTS5."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=120)
    conn.row_factory = sqlite3.Row
    
    sql = """
        SELECT l.gbif_id, l.title, l.abstract, l.keywords, l.topics,
               l.year, l.source, l.doi, l.language
        FROM literature l
        WHERE literature MATCH ?
          AND l.year >= ?
          AND l.language = 'eng'
          AND l.abstract IS NOT NULL AND l.abstract != ''
        LIMIT ?
    """
    
    results = []
    for row in conn.execute(sql, (query, year_min, limit)).fetchall():
        d = dict(row)
        d["pmid"] = f"gbif:{d.get('gbif_id', '?')}"
        d["journal"] = d.get("source", "")
        d["pub_year"] = d.get("year", "")
        results.append(d)
    
    conn.close()
    return results


def fetch_topic(args):
    query, label, limit = args
    try:
        papers = search_gbif(query, limit=limit)
        return (label, papers, None)
    except Exception as e:
        return (label, [], str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Max papers per topic")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent queries")
    parser.add_argument("--output", default="gbif_papers.jsonl")
    args = parser.parse_args()
    
    if not os.path.exists(DB_PATH):
        sys.exit(f"ERROR: GBIF FTS5 not found at {DB_PATH}")
    
    print(f"GBIF HARVEST: {len(TOPICS)} topics × {args.limit} papers, {args.workers} workers")
    print(f"Database: {DB_PATH} ({os.path.getsize(DB_PATH)/1e9:.1f} GB)")
    print()
    
    tasks = [(q, l, args.limit) for q, l in TOPICS]
    topic_results = {}
    errors = []
    t_start = time.time()
    
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_topic, t): t[1] for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            label = futures[fut]
            try:
                lbl, papers, err = fut.result()
                if err:
                    errors.append((lbl, err))
                    print(f"  [{i+1}/{len(TOPICS)}] {lbl:25s} ERROR: {err[:60]}")
                else:
                    topic_results[lbl] = papers
                    print(f"  [{i+1}/{len(TOPICS)}] {lbl:25s} → {len(papers):5d} papers")
            except Exception as e:
                errors.append((label, str(e)))
    
    elapsed = time.time() - t_start
    
    # Deduplicate
    seen = set()
    all_papers = []
    topic_final = {}
    for label, papers in topic_results.items():
        new = []
        for p in papers:
            key = p.get("pmid", p.get("gbif_id", ""))
            if key not in seen:
                seen.add(key)
                p["topic"] = label
                new.append(p)
        topic_final[label] = len(new)
        all_papers.extend(new)
    
    with open(args.output, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"GBIF HARVEST COMPLETE in {elapsed:.1f}s")
    print(f"  Total:  {len(all_papers)} unique papers")
    print(f"  Output: {args.output}")
    print(f"  Rate:   {len(all_papers)/max(elapsed,1)*60:.0f} papers/min")
    print(f"\n  Top 10 topics:")
    for label, count in sorted(topic_final.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(50, count // 10)
        print(f"    {label:25s} {count:5d} {bar}")


if __name__ == "__main__":
    main()
