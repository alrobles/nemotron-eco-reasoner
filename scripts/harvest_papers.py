#!/usr/bin/env python3
"""
harvest_papers.py — Parallel PubMed FTS5 paper harvest across ecological subdomains.
Queries multiple topics concurrently using ThreadPoolExecutor.

Usage:
    python3 harvest_papers.py --limit 500 --workers 8 --output eco_papers.jsonl

Requires: PubMed FTS5 index at /home/a474r867/work/pubmed/index/pubmed_fts.db
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

DB_PATH = "/home/a474r867/work/pubmed/index/pubmed_fts.db"

TOPICS = [
    # Population & community ecology
    ("population dynamics density dependence regulation", "pop_dynamics"),
    ("community assembly functional diversity trait convergence", "community"),
    ("trophic cascade food web network stability interaction", "foodweb"),
    
    # Conservation & biodiversity
    ("conservation planning protected area prioritization systematic", "conservation"),
    ("extinction risk IUCN Red List threatened assessment", "extinction"),
    ("biodiversity monitoring indicator essential variable sampling", "biodiv_monitor"),
    
    # Movement & spatial ecology
    ("animal movement telemetry home range step selection function", "movement"),
    ("dispersal kernel seed dispersal connectivity corridor landscape", "dispersal"),
    ("spatial point pattern analysis Ripley K inhomogeneous", "spatial"),
    
    # Climate change
    ("climate change range shift phenological mismatch adaptation", "climate"),
    ("thermal tolerance physiological limit acclimation heat stress", "thermal"),
    ("climate velocity refugia microclimate buffering", "microclimate"),
    
    # Phylogenetic ecology
    ("phylogenetic comparative method trait evolution Brownian Ornstein", "phylo"),
    ("phylogenetic diversity community structure phylogenetic signal", "phylo_div"),
    
    # Macroecology
    ("species richness latitudinal diversity gradient macroecological", "macroeco"),
    ("species abundance distribution metabolic theory body size", "metabolic"),
    ("beta diversity turnover nestedness distance decay similarity", "beta_div"),
    
    # Ecological statistics & methods
    ("occupancy model detection probability imperfect sampling site", "occupancy"),
    ("integrated population model mark recapture state space", "ipm"),
    ("joint species distribution model hierarchical Bayesian latent", "jsdm"),
    ("generalized additive model spline ecological nonlinear smooth", "gam"),
    
    # Invasion biology
    ("invasive species impact native community biotic resistance enemy", "invasion"),
    ("introduced species naturalization invasion hypothesis propagule", "intro"),
    
    # Ecosystem ecology
    ("ecosystem function biodiversity experiment productivity stability", "ecosystem"),
    ("carbon sequestration forest biomass LiDAR remote sensing", "carbon"),
    ("nutrient cycling decomposition litter stoichiometry microbial", "nutrient"),
    
    # Disease ecology
    ("disease ecology zoonotic spillover dilution effect biodiversity", "disease"),
    ("host parasite coevolution virulence transmission evolution", "parasite"),
    ("vector borne disease climate land use change ecological", "vector"),
    
    # Urban & human ecology
    ("urbanization biodiversity homogenization gradient land cover", "urban"),
    ("ecosystem service valuation natural capital nature contribution", "ecoservice"),
    
    # GBIF / biodiversity informatics
    ("species occurrence data GBIF sampling bias correction citizen", "gbif"),
    ("biodiversity informatics data quality completeness uncertainty", "informatics"),
    ("citizen science iNaturalist eBird observer effort detection", "citizen_sci"),
    ("species distribution model MaxEnt overfitting spatial autocorrelation", "sdm_methods"),
    
    # Freshwater & marine
    ("stream macroinvertebrate bioassessment water quality index", "freshwater"),
    ("marine biodiversity coral reef fish biomass pelagic", "marine"),
    
    # Pollination & plant ecology
    ("pollination network plant pollinator mutualism specialization", "pollination"),
    ("plant functional trait leaf economic spectrum wood density", "plant_traits"),
]


def search_pubmed(query: str, limit: int = 500, year_min: int = 2015) -> list[dict]:
    """Query PubMed FTS5. Thread-safe: opens its own connection."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=120)
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


def fetch_topic(args):
    """Fetch papers for one topic. Returns (label, papers, error)."""
    query, label, limit = args
    try:
        papers = search_pubmed(query, limit=limit)
        return (label, papers, None)
    except Exception as e:
        return (label, [], str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Max papers per topic")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent FTS5 queries")
    parser.add_argument("--output", default="eco_papers_extra.jsonl")
    parser.add_argument("--topics", help="Comma-separated topic labels to filter")
    args = parser.parse_args()
    
    if not os.path.exists(DB_PATH):
        sys.exit(f"ERROR: FTS5 index not found at {DB_PATH}")
    
    # Filter topics if specified
    topics = TOPICS
    if args.topics:
        wanted = set(args.topics.split(","))
        topics = [(q, l) for q, l in TOPICS if l in wanted]
    
    print(f"PARALLEL HARVEST: {len(topics)} topics × {args.limit} papers, {args.workers} workers")
    print(f"Database: {DB_PATH} ({os.path.getsize(DB_PATH)/1e9:.1f} GB)")
    print()
    
    tasks = [(q, l, args.limit) for q, l in topics]
    topic_results = {}  # label -> list of papers
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
                    print(f"  [{i+1}/{len(topics)}] {lbl:25s} ERROR: {err[:60]}")
                else:
                    topic_results[lbl] = papers
                    print(f"  [{i+1}/{len(topics)}] {lbl:25s} → {len(papers):5d} papers")
            except Exception as e:
                errors.append((label, str(e)))
                print(f"  [{i+1}/{len(topics)}] {label:25s} CRASH: {e}")
    
    elapsed = time.time() - t_start
    
    # ── Deduplicate across topics ──
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
    
    # ── Write output ──
    with open(args.output, "w") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    # ── Report ──
    print(f"\n{'='*60}")
    print(f"HARVEST COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Topics:       {len(topics)} ({len(topic_results)} ok, {len(errors)} failed)")
    print(f"  Total papers: {len(all_papers)} unique (from {sum(len(v) for v in topic_results.values())} raw)")
    print(f"  Output:       {args.output}")
    print(f"  Rate:         {len(all_papers)/max(elapsed,1)*60:.0f} papers/min")
    
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for lbl, err in errors:
            print(f"    {lbl}: {err[:80]}")
    
    print(f"\n  Top 10 topics:")
    for label, count in sorted(topic_final.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(50, count // 10)
        print(f"    {label:25s} {count:5d} {bar}")


if __name__ == "__main__":
    main()
