#!/usr/bin/env python3
"""
ingest_s2orc.py — Filter S2ORC index for ecological papers and extract text.
Reads paper_locations.txt.gz from stdin, filters by ecological keywords,
downloads full JSON for matching papers, extracts abstract+body, saves JSONL.

Usage (with Slurm):
    zcat paper_locations.txt.gz | python3 ingest_s2orc.py --output s2orc_papers.jsonl --limit 10000

S2ORC index: ~50GB compressed, ~8.1M papers
Each matching paper: download ~1-2MB JSON, extract ~1-10KB text
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from typing import Optional

# Ecological keywords for filtering (case-insensitive)
ECO_KEYWORDS = [
    "ecology", "ecological", "biodiversity", "conservation", "species richness",
    "ecosystem service", "climate change", "habitat", "species distribution",
    "population dynamics", "community ecology", "macroecology", "biogeography",
    "invasion biology", "urban ecology", "disease ecology", "restoration ecology",
    "tropical forest", "marine ecology", "freshwater ecology", "soil ecology",
    "pollination", "seed dispersal", "functional trait", "phylogenetic",
    "extinction risk", "protected area", "deforestation", "land use change",
    "nitrogen deposition", "carbon sequestration", "phenology", "range shift",
    "niche model", "occupancy model", "mark-recapture", "remote sensing",
    "species distribution model", "SDM", "MaxEnt", "generalized additive model",
    "GLMM", "generalized linear mixed", "beta diversity", "alpha diversity",
]

# Also check subject/category
ECO_SUBJECTS = [
    "ecology", "environmental science", "biology", "zoology",
    "botany", "conservation", "biodiversity", "evolutionary biology",
    "geography", "oceanography", "atmospheric science", "agriculture",
    "forestry", "fisheries", "wildlife", "sustainability",
]


def is_ecological(paper: dict) -> bool:
    """Check if a paper is ecology-related based on metadata."""
    text = " ".join(str(v).lower() for v in paper.values() if isinstance(v, str))
    
    # Keyword matching
    kw_hits = 0
    for kw in ECO_KEYWORDS:
        if kw.lower() in text:
            kw_hits += 1
            if kw_hits >= 2:
                return True
    
    # Subject matching
    subjects = str(paper.get("subjects", "")).lower()
    for subj in ECO_SUBJECTS:
        if subj in subjects:
            return True
    
    return False


def download_paper(paper_id: str, max_retries: int = 3) -> Optional[dict]:
    """Download full S2ORC paper JSON from PDF Parse data."""
    # S2ORC PDF Parse stored on S3
    url = f"https://ai2-s2-research.s3.amazonaws.com/s2orc/2023-11/full/papers/{paper_id}.json.gz"
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                import gzip
                data = json.loads(gzip.decompress(resp.read()))
                return data
        except urllib.error.HTTPError as e:
            if e.code == 403 or e.code == 404:
                return None  # Paper not available
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
    return None


def extract_text(paper: dict) -> dict:
    """Extract abstract and body text from S2ORC paper JSON."""
    abstract = ""
    body = ""
    
    # Try abstract field
    if "abstract" in paper:
        abstract = str(paper["abstract"])[:2000]
    
    # Try body text from parsed sections
    if "body_text" in paper:
        sections = paper["body_text"]
        if isinstance(sections, list):
            body = " ".join(
                s.get("text", "") for s in sections 
                if isinstance(s, dict) and s.get("text")
            )[:5000]
    
    # Fallback to pdf_parse
    if not abstract and "pdf_parse" in paper:
        pp = paper["pdf_parse"]
        if isinstance(pp, dict):
            abstract = str(pp.get("abstract", ""))[:2000]
            if "body_text" in pp and isinstance(pp["body_text"], list):
                body = " ".join(
                    s.get("text", "") for s in pp["body_text"]
                    if isinstance(s, dict) and s.get("text")
                )[:5000]
    
    return {
        "abstract": abstract,
        "body": body,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="s2orc_papers.jsonl")
    parser.add_argument("--limit", type=int, default=10000, help="Max papers to download")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between downloads")
    args = parser.parse_args()
    
    print(f"S2ORC INGEST: filtering for ecology, max {args.limit} papers", flush=True)
    
    total_scanned = 0
    matched = 0
    downloaded = 0
    saved = 0
    
    out_f = open(args.output, "w", encoding="utf-8")
    
    for line in sys.stdin:
        total_scanned += 1
        
        if total_scanned % 100000 == 0:
            print(f"  Scanned: {total_scanned:,}, matched: {matched}, done: {downloaded}", flush=True)
        
        try:
            meta = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        
        if not is_ecological(meta):
            continue
        
        matched += 1
        
        paper_id = meta.get("paper_id", "")
        if not paper_id:
            continue
        
        # Check if we already have this paper
        if downloaded >= args.limit:
            break
        
        # Download full paper
        full = download_paper(paper_id)
        if full is None:
            continue
        
        downloaded += 1
        
        # Extract text
        text = extract_text(full)
        abstract = text["abstract"]
        body = text["body"]
        
        # Skip papers without meaningful abstract
        if len(abstract) < 100:
            continue
        
        # Build our format
        title = meta.get("title", full.get("title", "Unknown"))
        year = meta.get("year", full.get("year", ""))
        authors = meta.get("authors", "")
        if isinstance(authors, list):
            authors = "; ".join(a[:3] for a in authors)
        
        rec = {
            "pmid": f"s2orc:{paper_id}",
            "title": str(title)[:500],
            "abstract": abstract,
            "authors": str(authors)[:500],
            "journal": str(meta.get("journal", full.get("journal", "S2ORC")))[:200],
            "pub_year": str(year)[:4],
            "keywords": str(meta.get("subjects", ""))[:300],
            "mesh_terms": "s2orc:ecology",
            "body": body[:5000],  # For CoT generation with full context
        }
        
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        saved += 1
        
        if saved % 100 == 0:
            print(f"  Saved: {saved} papers ({downloaded} downloaded)", flush=True)
        
        if saved >= args.limit:
            break
        
        time.sleep(args.delay)
    
    out_f.close()
    
    print(f"\n{'='*50}")
    print(f"S2ORC INGEST COMPLETE")
    print(f"  Scanned:    {total_scanned:,} papers")
    print(f"  Matched:    {matched:,} ecological")
    print(f"  Downloaded: {downloaded:,} full text")
    print(f"  Saved:      {saved:,} papers with abstract")
    print(f"  Output:     {args.output}")


if __name__ == "__main__":
    main()
