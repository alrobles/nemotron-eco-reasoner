#!/usr/bin/env python3
"""
harvest_papers.py — Query the PubMed FTS5 index for ecological papers
across multiple subdomains, deduplicate, and save as JSONL.

Extends the paper pool beyond SDM (4,402 papers) to cover:
- Population & community ecology
- Conservation & biodiversity  
- Movement & spatial ecology
- Climate change impacts
- Phylogenetic ecology
- Macroecology & biogeography
- Ecological statistics & methods
- Invasion biology
- Ecosystem ecology
- Disease ecology
- Urban ecology
- GBIF / biodiversity informatics

Usage:
    python3 harvest_papers.py --limit 500 --output new_papers.jsonl

Requires the PubMed FTS5 index at /home/a474r867/work/pubmed/index/pubmed_fts.db
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = "/home/a474r867/work/pubmed/index/pubmed_fts.db"

TOPICS = [
    # Population & community ecology
    ("population dynamics density dependence", "pop_dynamics"),
    ("community assembly functional diversity coexistence", "community"),
    ("trophic cascade food web network stability", "foodweb"),
    
    # Conservation & biodiversity
    ("conservation planning protected area prioritization Zonation", "conservation"),
    ("extinction risk IUCN Red List threatened species", "extinction"),
    ("biodiversity monitoring indicator essential biodiversity variable", "biodiv_monitor"),
    
    # Movement & spatial ecology  
    ("animal movement telemetry home range step selection", "movement"),
    ("dispersal kernel seed dispersal connectivity landscape", "dispersal"),
    ("spatial ecology point pattern analysis Ripley", "spatial"),
    
    # Climate change
    ("climate change range shift phenology mismatch", "climate"),
    ("thermal tolerance physiological limit acclimation", "thermal"),
    
    # Phylogenetic ecology
    ("phylogenetic comparative method trait evolution PGLS", "phylo"),
    ("phylogenetic diversity community phylogenetics", "phylo_div"),
    
    # Macroecology
    ("species richness latitudinal gradient macroecological pattern", "macroeco"),
    ("species abundance distribution metabolic theory ecology", "metabolic"),
    
    # Ecological statistics
    ("occupancy detection probability hierarchical model", "occupancy"),
    ("integrated population model mark recapture", "ipm"),
    ("joint species distribution model hierarchical Bayesian", "jsdm"),
    ("generalized additive model ecological nonlinear", "gam"),
    
    # Invasion biology
    ("invasive species impact native community biotic resistance", "invasion"),
    ("introduced species naturalization invasion hypothesis", "intro"),
    
    # Ecosystem ecology
    ("ecosystem function biodiversity experiment productivity", "ecosystem"),
    ("carbon sequestration forest biomass remote sensing", "carbon"),
    ("nutrient cycling decomposition litter soil ecology", "nutrient"),
    
    # Disease ecology
    ("disease ecology zoonotic spillover dilution effect biodiversity", "disease"),
    ("host parasite coevolution transmission network", "parasite"),
    
    # Urban ecology
    ("urbanization biodiversity homogenization gradient", "urban"),
    
    # GBIF / biodiversity informatics
    ("species occurrence data GBIF sampling bias citizen science", "gbif"),
    ("biodiversity informatics data quality completeness", "informatics"),
    ("citizen science iNaturalist eBird data integration", "citizen_sci"),
    ("species distribution model MaxEnt overfitting spatial autocorrelation", "sdm_methods"),
]


def search_pubmed(query: str, limit: int = 500, year_min: int = 2015) -> list[dict]:
    """Query PubMed FTS5 and return papers with metadata."""
    if not os.path.exists(DB_PATH):
        sys.exit(f"FTS index not found: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row
    
    sql = """
        SELECT a.pmid, a.title, a.abstract, a.journal,
               a.pub_year, a.authors, a.mesh_terms, a.keywords
        FROM articles_fts f
        JOIN articles a ON a.pmid = f.rowid
        WHERE articles_fts MATCH ?
          AND a.pub_year >= ?
          AND a.language = 'eng'
          AND a.abstract IS NOT NULL AND a.abstract != ''
        LIMIT ?
    """
    
    results = [dict(row) for row in conn.execute(sql, (query, year_min, limit)).fetchall()]
    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Max papers per topic")
    parser.add_argument("--output", default="new_papers.jsonl")
    parser.add_argument("--topics", help="Comma-separated topic labels (by index or name)")
    args = parser.parse_args()
    
    if not os.path.exists(DB_PATH):
        sys.exit(f"ERROR: FTS5 index not found at {DB_PATH}")
    
    print(f"Harvesting papers from {len(TOPICS)} ecological topics...")
    print(f"Max {args.limit} papers per topic, year >= 2015")
    
    seen_pmids = set()
    all_papers = []
    topic_counts = {}
    
    for i, (query, label) in enumerate(TOPICS):
        print(f"  [{i+1}/{len(TOPICS)}] {label}: '{query[:60]}...'", end=" ", flush=True)
        try:
            papers = search_pubmed(query, limit=args.limit)
            new = [p for p in papers if p["pmid"] not in seen_pmids]
            for p in new:
                seen_pmids.add(p["pmid"])
                p["topic"] = label
            all_papers.extend(new)
            topic_counts[label] = len(new)
            print(f"→ {len(new)} new (total unique: {len(seen_pmids)})")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.2)  # Gentle on the DB
    
    # Save
    with open(args.output, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*50}")
    print(f"TOTAL: {len(all_papers)} unique papers ({len(seen_pmids)} total)")
    print(f"Output: {args.output}")
    print(f"\nBy topic:")
    for label, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:20s} {count:5d} papers")


if __name__ == "__main__":
    main()
