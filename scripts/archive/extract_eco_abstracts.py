#!/usr/bin/env python3
"""
Extract ecological paper abstracts from PubMed FTS5 index.
Outputs JSON with title, abstract for downstream Q&A generation.

Usage (on cluster):
    python3 extract_eco_abstracts.py --output eco_abstracts.json --limit 500
"""
import argparse
import json
import os
import sys
import sqlite3
import random

random.seed(42)

DB_PATH = "/home/a474r867/work/pubmed/index/pubmed_fts.db"

ECO_TOPICS = [
    "species distribution model ecological niche MaxEnt",
    "species range shift climate change distribution",
    "ecological niche model habitat suitability biodiversity",
    "species coexistence competition niche partitioning community",
    "community assembly trait-based ecology functional diversity",
    "ecological network food web trophic interaction",
    "host parasite coevolution disease ecology",
    "disease ecology zoonotic spillover biodiversity",
    "macroecological pattern species richness latitude gradient",
    "biogeographic region species turnover beta diversity",
    "island biogeography species-area relationship extinction",
    "conservation prioritization biodiversity hotspot protected area",
    "extinction risk assessment IUCN ecological traits",
    "invasive species impact native community ecosystem",
    "phylogenetic comparative method ecological trait evolution",
    "joint species distribution model hierarchical Bayesian",
    "occupancy model detection probability imperfect sampling",
    "phenological shift climate warming ecological mismatch",
    "thermal tolerance physiological limit range edge",
    "ecosystem service biodiversity function resilience",
    "land use change habitat fragmentation connectivity",
    "marine biogeography larval dispersal connectivity",
    "freshwater ecology river network metacommunity",
    "plant-pollinator network mutualism coevolution",
    "tropical forest biodiversity deforestation fragmentation",
]


def search_pubmed(query: str, limit: int = 20) -> list[dict]:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT pmid, doi, title, abstract, journal, pub_year, authors, mesh_terms
        FROM articles_fts f
        JOIN articles a ON a.pmid = f.rowid
        WHERE articles_fts MATCH ?
        AND a.language = 'eng'
        AND a.abstract IS NOT NULL
        AND length(a.abstract) > 500
        ORDER BY rank LIMIT ?
    """
    rows = conn.execute(sql, [query, limit]).fetchall()
    results = [dict(r) for r in rows]
    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eco_abstracts.json")
    parser.add_argument("--limit", type=int, default=500,
                        help="Total abstracts to extract (across all topics)")
    parser.add_argument("--per-topic", type=int, default=30,
                        help="Max abstracts per topic")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"ERROR: PubMed FTS5 not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    all_abstracts = []
    seen_pmids = set()
    
    print(f"Querying {len(ECO_TOPICS)} ecological topics...")
    for i, topic in enumerate(ECO_TOPICS):
        if len(all_abstracts) >= args.limit:
            break
        try:
            results = search_pubmed(topic, limit=args.per_topic)
            new = 0
            for r in results:
                pmid = str(r["pmid"])
                if pmid in seen_pmids:
                    continue
                if len(r.get("abstract", "")) < 500:
                    continue
                seen_pmids.add(pmid)
                all_abstracts.append({
                    "pmid": pmid,
                    "title": r.get("title", ""),
                    "abstract": r.get("abstract", ""),
                    "journal": r.get("journal", ""),
                    "pub_year": r.get("pub_year", ""),
                    "authors": r.get("authors", ""),
                    "mesh_terms": r.get("mesh_terms", ""),
                    "topic": topic,
                })
                new += 1
            print(f"  [{i+1}/{len(ECO_TOPICS)}] '{topic[:50]}...' → {new} new (total: {len(all_abstracts)})")
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            continue

    random.shuffle(all_abstracts)
    all_abstracts = all_abstracts[:args.limit]

    with open(args.output, "w") as f:
        json.dump(all_abstracts, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {len(all_abstracts)} abstracts → {args.output}")
    total_chars = sum(len(a["abstract"]) for a in all_abstracts)
    print(f"Total abstract text: {total_chars:,} chars (~{total_chars/4:.0f} tokens)")


if __name__ == "__main__":
    main()
