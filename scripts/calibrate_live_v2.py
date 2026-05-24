#!/usr/bin/env python3
"""
Research Calibration Script v2 — Multi-Needle Enrichment Protocol
=================================================================
Purpose: Empirically derive fidelity (f) and decay rate (d) for the research paper.
         Uses 5-needle injection per run for efficient decay curve measurement.

Models:  G2.5-FLASH, G3.0-FLASH, G3.1-FLASH-LITE, G2.5-PRO, G3.0-PRO
Depths:  5,000 / 10,000 / 50,000 / 100,000 turns
Target:  N=30 per depth per model (configurable via --target-n)

Needle Design:
  Each run injects 5 needles at depths 5%, 20%, 50%, 80%, 95% of the haystack.
  This produces 5 recall observations at different "ages", enabling curve-fitting
  of P(E) = f * (1 - age * d).

Usage:
  python3 calibrate_live_v2.py --dry-run                    # Show plan + costs
  python3 calibrate_live_v2.py --model G2.5-FLASH           # Single model
  python3 calibrate_live_v2.py --flash-only                 # All Flash models
  python3 calibrate_live_v2.py --pro-only --target-n 10     # Pro pilot (N=10)
  python3 calibrate_live_v2.py                              # ALL models (default N=30)
"""

import os
import json
import time
import random
import sys
import glob
import argparse
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

sys.path.append(os.path.join(os.getcwd(), "scripts/benchmarking"))
from run_cst import CSTBenchmarker

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

FLASH_MODELS = {
    "G2.5-FLASH":      "models/gemini-2.5-flash",
    "G3.0-FLASH":      "models/gemini-3-flash-preview",
    "G3.1-FLASH-LITE": "models/gemini-3.1-flash-lite-preview",
}

PRO_MODELS = {
    "G2.5-PRO":  "models/gemini-2.5-pro",
    "G3.0-PRO":  "models/gemini-3-pro-preview",
}

ALL_MODELS = {**FLASH_MODELS, **PRO_MODELS}

TARGET_DEPTHS_FLASH = [5000, 10000, 40000]          # Up to ~1M tokens
TARGET_DEPTHS_PRO   = [5000, 10000, 40000, 80000]   # Up to ~2M tokens

# Needle injection positions (fraction of total turns)
NEEDLE_POSITIONS = [0.05, 0.20, 0.50, 0.80, 0.95]

API_KEYS = [k for k in [
    os.getenv("GOOGLE_API_KEY_CLOUD"),     # $300 Credit Account (Primary)
    os.getenv("GOOGLE_API_KEY"),           # Free Tier Fallback 1
    os.getenv("GOOGLE_API_KEY_FALLBACK"),  # Free Tier Fallback 2
] if k]

# Cooldown buffers (seconds) — respects TPM limits
COOLDOWN = {
    5000:   35,
    10000:  70,
    50000:  300,
    100000: 600,
}

# Pricing per 1M input tokens (Google AI Studio, March 2026)
PRICE_PER_1M_INPUT = {
    "G2.5-FLASH":      0.15,
    "G3.0-FLASH":      0.15,
    "G3.1-FLASH-LITE": 0.08,
    "G2.5-PRO":        1.25,
    "G3.0-PRO":        1.25,
}

# Output directory
OUTPUT_DIR = "test/benchmarks"

# ============================================================
# HAYSTACK & MULTI-NEEDLE TEST
# ============================================================

def generate_haystack_with_needles(turns_target):
    """Generate haystack with 5 needles at fixed depth positions.
    
    Returns:
        tuple: (full_prompt, needle_manifest)
        needle_manifest: list of dicts with 'id', 'value', 'turn', 'depth_frac'
    """
    random.seed(time.time())  # Cold-start: unique each run
    bench = CSTBenchmarker({})
    
    # Generate needle identifiers
    needles = []
    for i, pos in enumerate(NEEDLE_POSITIONS):
        turn = int(turns_target * pos)
        needle_id = f"NEEDLE_{chr(65+i)}_{random.randint(1000, 9999)}"
        needle_value = f"VALUE_{random.randint(10000, 99999)}"
        needles.append({
            "id": needle_id,
            "value": needle_value,
            "turn": turn,
            "depth_frac": pos,
            "age": turns_target - turn,  # How old this needle is at retrieval
        })
    
    # Build haystack with needles injected at correct positions
    needle_turns = {n["turn"]: n for n in needles}
    haystack_parts = []
    
    for i in range(turns_target):
        if i in needle_turns:
            n = needle_turns[i]
            haystack_parts.append(f"CRITICAL DATA POINT: {n['id']} is {n['value']}.")
        else:
            haystack_parts.append(bench.generate_realistic_noise(i))
    
    full_context = "\n".join(haystack_parts)
    
    # Build retrieval prompt asking for ALL needles
    needle_query = ", ".join([n["id"] for n in needles])
    prompt = (
        f"{full_context}\n\n"
        f"Question: What are the values of the following data points: {needle_query}? "
        f"Respond in JSON format: {{\"results\": [{{\"id\": \"...\", \"value\": \"...\"}}]}}"
    )
    
    return prompt, needles


def run_single_test(api_key, turns_size, model_api_id, model_label, key_index):
    """Execute a single multi-needle test via REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_api_id}:generateContent?key={api_key}"
    
    prompt, needles = generate_haystack_with_needles(turns_size)
    effective_tokens = turns_size * 25
    timestamp = time.time()
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0,  # Deterministic for reproducibility
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=max(180, turns_size // 30))
        latency = time.time() - start_time
        
        if response.status_code == 200:
            res_json = response.json()
            try:
                text_out = res_json["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(text_out)
                
                # Score each needle individually
                retrieved = {}
                if "results" in data:
                    for item in data["results"]:
                        retrieved[item.get("id", "")] = item.get("value", "")
                
                needle_results = []
                for n in needles:
                    recalled = (retrieved.get(n["id"]) == n["value"])
                    needle_results.append({
                        "needle_id": n["id"],
                        "expected": n["value"],
                        "got": retrieved.get(n["id"], "MISSING"),
                        "recalled": recalled,
                        "depth_frac": n["depth_frac"],
                        "age_turns": n["age"],
                    })
                
                overall_success = all(nr["recalled"] for nr in needle_results)
                recall_count = sum(1 for nr in needle_results if nr["recalled"])
                
                return {
                    "success": overall_success,
                    "recall_count": recall_count,
                    "recall_total": len(needles),
                    "recall_rate": recall_count / len(needles),
                    "needle_results": needle_results,
                    "latency": latency,
                    "status": 200,
                    "turns": turns_size,
                    "tokens": effective_tokens,
                    "model": model_label,
                    "timestamp": timestamp,
                    "key_index": key_index,
                    "iso_time": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                return {
                    "success": False,
                    "status": "parse_err",
                    "error": str(e)[:200],
                    "turns": turns_size,
                    "tokens": effective_tokens,
                    "model": model_label,
                    "timestamp": timestamp,
                    "key_index": key_index,
                }
        else:
            return {
                "success": False,
                "status": response.status_code,
                "error": response.text[:200],
                "turns": turns_size,
                "tokens": effective_tokens,
                "model": model_label,
                "timestamp": timestamp,
                "key_index": key_index,
            }
    except Exception as e:
        return {
            "success": False,
            "status": "exception",
            "error": str(e)[:200],
            "turns": turns_size,
            "model": model_label,
            "timestamp": timestamp,
            "key_index": key_index,
        }


# ============================================================
# DATA MANAGEMENT
# ============================================================

def get_output_path(model_key, depth):
    """Output file per model+depth for clean separation."""
    return os.path.join(OUTPUT_DIR, f"live_calibration_{model_key}_{depth}_v2.json")


def load_existing_success_count(model_key, depth):
    """Count ALL existing successful samples for a model+depth across all files."""
    count = 0
    
    # Count from legacy files (any pattern matching this model)
    legacy_pattern = os.path.join(OUTPUT_DIR, f"live_calibration_{model_key}_*.json")
    for f in glob.glob(legacy_pattern):
        try:
            for s in json.load(open(f)):
                if s.get("success") and s.get("turns") == depth:
                    count += 1
        except Exception:
            pass
    
    return count


def load_enrichment_file(model_key, depth):
    """Load existing enrichment samples from the v2 output file."""
    path = get_output_path(model_key, depth)
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return []


def save_enrichment_file(model_key, depth, results):
    """Save enrichment results to the v2 output file."""
    path = get_output_path(model_key, depth)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ============================================================
# PLANNING & COST
# ============================================================

def build_enrichment_plan(model_key, target_n):
    """Calculate remaining samples needed per depth."""
    plan = []
    
    # Select appropriate depths based on the model's physical window
    target_depths = TARGET_DEPTHS_FLASH if "FLASH" in model_key else TARGET_DEPTHS_PRO
    
    for depth in target_depths:
        # Count from ALL sources (legacy + enrichment v2)
        legacy_count = load_existing_success_count(model_key, depth)
        # Subtract enrichment v2 count from legacy (it's already included via glob)
        enrichment_data = load_enrichment_file(model_key, depth)
        enrichment_success = sum(1 for s in enrichment_data if s.get("success"))
        
        total_have = legacy_count  # legacy glob already includes v2 file
        need = max(0, target_n - total_have)
        
        tokens = depth * 25
        price = PRICE_PER_1M_INPUT.get(model_key, 0.15)
        cost_per_run = (tokens / 1_000_000) * price
        
        plan.append({
            "depth": depth,
            "tokens": tokens,
            "have": total_have,
            "need": need,
            "cost_per_run": cost_per_run,
            "tier_cost": cost_per_run * need,
        })
    return plan


# ============================================================
# EXECUTION ENGINE
# ============================================================

def run_enrichment(model_key, model_api_id, plan, dry_run=False):
    """Execute enrichment for a single model."""
    total_runs = 0
    key_idx = 0
    consecutive_429s = 0
    
    for tier in plan:
        depth = tier["depth"]
        remaining = tier["need"]
        
        if remaining == 0:
            print(f"  ✅ {depth:>7,} turns: Already at target (have={tier['have']})")
            continue
        
        print(f"\n  🔬 {depth:>7,} turns: Running {remaining} samples "
              f"(have={tier['have']}, target={tier['have'] + remaining})...")
        
        if dry_run:
            continue
        
        # Load existing enrichment data for this depth
        enrichment_results = load_enrichment_file(model_key, depth)
        
        for i in range(remaining):
            if not API_KEYS:
                print("    ❌ No API keys configured!")
                return enrichment_results, total_runs
            
            current_key = API_KEYS[key_idx % len(API_KEYS)]
            cooldown = COOLDOWN.get(depth, 60)
            
            print(f"    [{i+1}/{remaining}] {depth:,} turns, Key {key_idx % len(API_KEYS)}, "
                  f"5 needles...", end="", flush=True)
            
            # Cooldown between runs
            if i > 0 or total_runs > 0:
                time.sleep(cooldown)
            
            result = run_single_test(current_key, depth, model_api_id, model_key, 
                                     key_idx % len(API_KEYS))
            enrichment_results.append(result)
            total_runs += 1
            
            # Save immediately for durability
            save_enrichment_file(model_key, depth, enrichment_results)
            
            status = result.get("status")
            if status == 200:
                consecutive_429s = 0
                rc = result.get("recall_count", 0)
                rt = result.get("recall_total", 5)
                lat = result.get("latency", 0)
                print(f" {rc}/{rt} recalled ({lat:.2f}s)")
            elif status == 429:
                consecutive_429s += 1
                print(f" 429 — switching key (streak: {consecutive_429s})")
                key_idx += 1
                
                # Exponential Backoff Contingency Plan
                if consecutive_429s >= len(API_KEYS):
                    backoff_tier = (consecutive_429s // len(API_KEYS))
                    if backoff_tier == 1:
                        wait_time = 60  # 1 min
                    elif backoff_tier == 2:
                        wait_time = 300 # 5 mins
                    elif backoff_tier == 3:
                        wait_time = 900 # 15 mins
                    else:
                        print(f"\n    ⛔ SUSTAINED 429 QUOTA BLOCK. All keys exhausted and backoffs failed.")
                        print(f"       ✅ Progress safely saved to JSON. Exiting script to prevent spam.")
                        print(f"       🔄 You can restart this exact command later and it will resume perfectly.")
                        return enrichment_results, total_runs
                        
                    print(f"\n    ⚠️ QUOTA WARNING: All keys hit 429.")
                    print(f"       ⏳ Backoff Tier {backoff_tier}: Pausing script for {wait_time/60:.0f} minutes...")
                    time.sleep(wait_time)
                    print(f"       ▶️ Resuming...")
            else:
                print(f" ERROR ({status})")
    
    return None, total_runs


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Research Calibration v2 — Multi-Needle Enrichment Protocol")
    parser.add_argument("--model", type=str, choices=list(ALL_MODELS.keys()),
                        help="Run a single model")
    parser.add_argument("--flash-only", action="store_true",
                        help="Run Flash models only")
    parser.add_argument("--pro-only", action="store_true",
                        help="Run Pro models only")
    parser.add_argument("--target-n", type=int, default=30,
                        help="Target N per depth tier (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan and costs without executing")
    args = parser.parse_args()
    
    # Determine which models to run
    if args.model:
        models_to_run = {args.model: ALL_MODELS[args.model]}
    elif args.flash_only:
        models_to_run = FLASH_MODELS
    elif args.pro_only:
        models_to_run = PRO_MODELS
    else:
        models_to_run = ALL_MODELS
    
    target_n = args.target_n
    
    print("=" * 70)
    print("RESEARCH CALIBRATION v2 — MULTI-NEEDLE ENRICHMENT PROTOCOL")
    print(f"Target: N={target_n} per depth tier per model")
    print(f"Depths (Flash): {TARGET_DEPTHS_FLASH}")
    print(f"Depths (Pro): {TARGET_DEPTHS_PRO}")
    print(f"Needles per run: {len(NEEDLE_POSITIONS)} at positions {NEEDLE_POSITIONS}")
    print(f"Models: {list(models_to_run.keys())}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 70)
    
    grand_total_cost = 0.0
    grand_total_time = 0.0
    grand_total_runs = 0
    
    for model_key, model_api_id in models_to_run.items():
        print(f"\n{'─' * 70}")
        print(f"MODEL: {model_key} ({model_api_id})")
        print(f"{'─' * 70}")
        
        plan = build_enrichment_plan(model_key, target_n)
        cost = sum(t["tier_cost"] for t in plan)
        total_needed = sum(t["need"] for t in plan)
        est_time_min = sum(
            t["need"] * (COOLDOWN.get(t["depth"], 60) + 15)
            for t in plan if t["need"] > 0
        ) / 60
        
        grand_total_cost += cost
        grand_total_time += est_time_min
        grand_total_runs += total_needed
        
        # Print plan table
        print(f"\n  {'Depth':>10} {'Have':>6} {'Need':>6} {'Tokens':>10} "
              f"{'$/Run':>8} {'Tier $':>8}")
        print(f"  {'─'*10} {'─'*6} {'─'*6} {'─'*10} {'─'*8} {'─'*8}")
        for tier in plan:
            print(f"  {tier['depth']:>10,} {tier['have']:>6} {tier['need']:>6} "
                  f"{tier['tokens']:>10,} ${tier['cost_per_run']:>7.4f} "
                  f"${tier['tier_cost']:>7.2f}")
        print(f"\n  Model Total: {total_needed} runs, ${cost:.2f}, ~{est_time_min:.0f} min")
        
        if not args.dry_run and total_needed > 0:
            run_enrichment(model_key, model_api_id, plan, dry_run=False)
    
    # Grand Summary
    print(f"\n{'=' * 70}")
    print(f"GRAND TOTAL")
    print(f"  API Calls:      {grand_total_runs}")
    print(f"  Estimated Cost:  ${grand_total_cost:.2f}")
    print(f"  Estimated Time:  ~{grand_total_time:.0f} min ({grand_total_time/60:.1f} hours)")
    print(f"  Data Points:     {grand_total_runs * len(NEEDLE_POSITIONS)} "
          f"({grand_total_runs} runs × {len(NEEDLE_POSITIONS)} needles)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
