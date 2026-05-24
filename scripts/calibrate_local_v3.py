#!/usr/bin/env python3
"""
Research Calibration Script v3 — Local Model N=30 Protocol
==========================================================
Purpose: Empirically derive fidelity (f) and decay rate (d) for local models.
          Uses 5-needle injection per run for curve-fitting.

Models:  MISTRAL-22B (Ollama), QWEN-32B (MLX Brain)
Depths:  5,000 / 10,000 turns (Focus on agentic working memory)
Target:  N=30 per depth per model
"""

import os
import json
import time
import random
import sys
import argparse
import requests
from datetime import datetime, timezone

# Ensure local imports work
sys.path.append(os.path.join(os.getcwd(), "scripts/benchmarking"))
from run_cst import CSTBenchmarker

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MLX_URL = "http://localhost:8005/v1/completions"

LOCAL_MODELS = {
    "MISTRAL-22B": "mlx-community/Mistral-Small-Instruct-2409-4bit",
    "QWEN-32B":    "/Users/akashagrawal/PycharmProjects/moltbot/models/Qwen2.5-Coder-32B-Instruct-3bit",
}

TARGET_DEPTHS = [100, 500, 1000, 2500]
NEEDLE_POSITIONS = [0.05, 0.20, 0.50, 0.80, 0.95]
OUTPUT_DIR = "test/benchmarks"

# ============================================================
# HAYSTACK & MULTI-NEEDLE TEST
# ============================================================

def generate_haystack_with_needles(turns_target, mode="cold"):
    """Generate haystack with 5 needles.
    Mode 'warm' uses a deterministic seed (42) to enable KV-cache reuse.
    Mode 'cold' uses a random seed (time.time()) to force a re-ingest.
    """
    if mode == "warm":
        random.seed(42)
    else:
        random.seed(time.time())
        
    bench = CSTBenchmarker({})
    
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
            "age": turns_target - turn,
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
    needle_query = ", ".join([n["id"] for n in needles])
    # MLX-Brain prompt format (ChatML style for instructions)
    prompt = (
        f"<|im_start|>user\n{full_context}\n\n"
        f"Question: What are the values of the following data points: {needle_query}? "
        f"Respond ONLY in JSON format: {{\"results\": [{{\"id\": \"...\", \"value\": \"...\"}}]}}<|im_end|>\n"
        f"<|im_start|>assistant\n{{\"results\":"
    )
    
    return prompt, needles

def run_local_test(model_key, prompt, turns_size, context=None):
    """Execute test against MLX-Brain for maximum prefill speed."""
    start_time = time.time()
    
    # All local models now use MLX-Brain for 30x prefill advantage
    payload = {
        "model": LOCAL_MODELS[model_key],
        "prompt": prompt, 
        "max_tokens": 500, 
        "temperature": 0.0,
        "stop": ["<|im_end|>", "Question:"]
    }
    
    # Note: MLX-Brain handles KV-cache warmth internally via the server process
    # but we force 3600s for the massive 125k ingests
    response = requests.post(MLX_URL, json=payload, timeout=3600)
    res_json = response.json()
    
    # Ensure raw output includes the JSON prefix if omitted by model
    text_out = res_json["choices"][0]["text"]
    print(f"      [DEBUG] Raw output: {text_out}")
    if not text_out.strip().startswith("["):
        text_out = "{\"results\":" + text_out
        
    latency = time.time() - start_time
    return text_out, latency, None

def score_response(text_out, needles):
    text_out = text_out.strip()
    
    # Handle the prefix/suffix wrapper for the forced JSON completion
    if text_out.startswith("["):
        # Model returned just the list, we wrap it
        text_out = "{\"results\":" + text_out
        
    if text_out.startswith("{\"results\":") and "]" in text_out:
        # Ensure it ends with the closing brace
        last_bracket = text_out.rfind("]")
        text_out = text_out[:last_bracket+1] + "}"
        
    # Final safety: truncate any 'Extra data' (hallucinated text after JSON)
    if "}" in text_out:
        text_out = text_out[:text_out.rfind("}")+1]
    elif "]" in text_out:
        text_out = text_out[:text_out.rfind("]")+1]
        
    try:
        data = json.loads(text_out)
        retrieved = {}
        if isinstance(data, dict) and "results" in data:
            for item in data["results"]:
                retrieved[str(item.get("id", ""))] = str(item.get("value", ""))
        elif isinstance(data, list):
            for item in data:
                retrieved[str(item.get("id", ""))] = str(item.get("value", ""))
        
        results = []
        for n in needles:
            recalled = (retrieved.get(str(n["id"])) == str(n["value"]))
            results.append({"id": n["id"], "recalled": recalled, "age": n["age"]})
        
        return results, True
    except Exception as e:
        print(f"      [ERROR] JSON Parse Failed: {e}")
        return [{"id": n["id"], "recalled": False, "age": n["age"]} for n in needles], False

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=LOCAL_MODELS.keys(), required=True)
    parser.add_argument("--target-n", type=int, default=30)
    parser.add_argument("--diagnostic", action="store_true", help="Run the 4-Quadrant Matrix (Warm/Cold x 5k/10k)")
    args = parser.parse_args()
    
    model_key = args.model
    target_n = args.target_n
    
    # Ramping depths to map the Discovery Cliff
    quadrants = []
    for d in TARGET_DEPTHS:
        quadrants.append((d, "warm"))
        quadrants.append((d, "cold"))
    
    if args.diagnostic:
        print(f"[*] Initializing 4-Quadrant Diagnostic Sweep for {model_key}...")
        target_n_quadrant = 5 # Rapid validation
        output_file = os.path.join(OUTPUT_DIR, f"live_calibration_{model_key}_diagnostic_v3.json")
    else:
        print(f"[*] Starting Full N=120 Calibration for {model_key} (30 per quadrant)...")
        target_n_quadrant = target_n
        output_file = os.path.join(OUTPUT_DIR, f"live_calibration_{model_key}_v3.json")
    
    all_results = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_results = json.load(f)
        print(f"  Loaded {len(all_results)} existing samples from {output_file}")

    for depth, mode in quadrants:
        # Filter existing results for this quadrant
        existing_count = sum(1 for r in all_results if r["depth"] == depth and r.get("mode", "cold") == mode and r["success"])
        needed = target_n_quadrant - existing_count
        
        if needed <= 0:
            print(f"  ✅ Quadrant ({depth}, {mode}) already completed.")
            continue

        last_context = None # Reset context for each quadrant
        print(f"  🔬 Testing quadrant ({depth} turns, {mode})...")
        for i in range(needed):
            print(f"    [{i+1}/{needed}]", end=" ", flush=True)
            prompt, needles = generate_haystack_with_needles(depth, mode=mode)
            
            try:
                # Optimized: Thread context for KV-cache reuse in warm mode
                current_context = last_context if mode == "warm" else None
                text_out, lat, new_context = run_local_test(model_key, prompt, depth, context=current_context)
                
                if mode == "warm":
                    last_context = new_context
                
                scores, success = score_response(text_out, needles)
                recalled_count = sum(1 for s in scores if s["recalled"])
                
                result = {
                    "model": model_key,
                    "depth": depth,
                    "mode": mode,
                    "iteration": existing_count + i,
                    "latency": lat,
                    "success": success,
                    "recall_count": recalled_count,
                    "recall_total": len(needles),
                    "recall_rate": recalled_count / len(needles),
                    "scores": scores,
                    "timestamp": time.time(),
                    "iso_time": datetime.now(timezone.utc).isoformat()
                }
                all_results.append(result)
                print(f"Done ({recalled_count}/{len(needles)} recalled, {lat:.2f}s)")
            except Exception as e:
                print(f"Error: {e}")
                
            # Periodic save
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"[*] Task complete for {model_key}. Results in {output_file}")

if __name__ == "__main__":
    main()
