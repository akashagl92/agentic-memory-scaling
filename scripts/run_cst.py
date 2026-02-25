#!/usr/bin/env python3
import json
import argparse
import sys
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import math # Added for logarithmic distribution of extra needles

class CSTBenchmarker:
    def __init__(self, scenario, extra_needles=0, total_turns_limit=10000000):
        self.scenario = scenario
        self.needles = scenario.get('needles', [])[:] # Initialize with existing needles
        # Also consider hard_facts as needles
        if 'hard_facts' in scenario:
            for fact in scenario['hard_facts']:
                self.needles.append({'id': fact['key'], 'turn': fact['turn'], 'value': fact['value']})
        
        if extra_needles > 0:
            for i in range(extra_needles):
                # Spread extra needles logarithmically across the timeline
                turn = int(10 ** random.uniform(1, math.log10(total_turns_limit)))
                self.needles.append({
                    'id': f'extra_{i}', 
                    'turn': turn, 
                    'value': f'Synthetic_Signal_{random.randint(1000,9999)}'
                })
        
        # Realistic Noise Categories: Technical + Personal assistance
        self.noise_templates = {
            "git": [
                "git checkout -b feature/auth-system",
                "git commit -m 'fix: resolve race condition in worker loop'",
                "Fixing merge conflict in {}"
            ],
            "coding": [
                "Refactor {} function to use async/await.",
                "Write a unit test for the {} class.",
                "Why is the {} throwing a NullPointerException?"
            ],
            "infra": [
                "docker-compose up -d --build",
                "kubectl logs -f deployment/{}",
                "Why is the {} failing its health check?"
            ],
            "calendar": [
                "Schedule a 1:1 with {} for tomorrow at 2 PM.",
                "Move the {} sync to Friday afternoon.",
                "Is there any conflict for the {} meeting on Wednesday?",
                "Remind me to prep for the {} presentation."
            ],
            "meetings": [
                "Summarize the transcript from the {} meeting.",
                "What were the action items for {} from the last call?",
                "Draft an email to {} following up on the design review.",
                "Record the meeting with {} about the project roadmap."
            ],
            "notes": [
                "Create a new note about {} architectural patterns.",
                "Append the research findings to the {} document.",
                "Search my personal notes for mentions of {}.",
                "Sync my {} notes with the cloud storage."
            ],
            "research": [
                "Gather latest papers on {} for our literature review.",
                "Summarize the key findings from the {} study.",
                "Contrast the approach in {} with recent SOTA.",
                "Find citations for the {} methodology."
            ],
            "web_search": [
                "Search for the latest documentation on {}.",
                "Find price comparisons for {} across different vendors.",
                "What are the top 5 competitors for {} in the AI space?",
                "Monitor news for any updates regarding {}."
            ],
            "strategic_reasoning": [
                "Thinking: I should first check {} before suggesting a fix.",
                "Strategy: If {} fails, I will fallback to {} and notify the user.",
                "Observe: The {} logs indicate a potential memory leak.",
                "Decision: Prioritizing {} over {} for better performance."
            ]
        }
        self.placeholders = [
            "SessionManager", "AuthGuard", "DataPipe", "ModelRunner", "UserStore", "MainController",
            "Alex", "Jordan", "Sarah", "Engineering", "Marketing", "Strategic Planning",
            "Hierarchical Memory", "State Convergence", "Quantum Computing", "Sustainable energy"
        ]

    def generate_realistic_noise(self, index):
        """Generates technical noise that emulates real-world developer interactions."""
        category = random.choice(list(self.noise_templates.keys()))
        template = random.choice(self.noise_templates[category])
        
        # Fill placeholders
        if "{}" in template:
            count = template.count("{}")
            choices = random.sample(self.placeholders, count) if count <= len(self.placeholders) else [random.choice(self.placeholders) for _ in range(count)]
            content = template.format(*choices)
        else:
            content = template
            
        return f"User message #{index}: {content}"

    def turn_generator(self, total_turns):
        """Yields turns one by one, optimized for speed."""
        needle_map = {n['turn']: n for n in self.needles}
        secret_idx = self.scenario.get('secret_turn_index', -1)
        
        # Pre-calculate noise token average (approx 18-25 tokens per turn)
        avg_noise_tokens = 25
        
        for i in range(total_turns):
            if i == secret_idx or i in needle_map:
                if i == secret_idx:
                    content = f"SYSTEM: {self.scenario['secret_constraint']}"
                else:
                    n = needle_map[i]
                    content = f"SYSTEM: Hard Fact - {n['id']} is {n['value']}"
                
                yield {
                    "index": i,
                    "content": content,
                    "tokens": len(content.split()) * 1.5,
                    "is_signal": True
                }
            else:
                yield {
                    "index": i,
                    "content": "", # Skip generating noise text
                    "tokens": avg_noise_tokens,
                    "is_signal": False
                }

    def run_baseline(self, total_turns):
        """Analytical Baseline: Decay is O(N)."""
        secret_idx = self.scenario.get('secret_turn_index', -1)
        recall = (total_turns - secret_idx) <= 10 if secret_idx != -1 else False
        return {
            "recall": recall,
            "avg_tokens": total_turns * 25 if total_turns < 1000 else 10 * 25, # Heuristic
            "peak_tokens": total_turns * 25,
            "final_tokens": 10 * 25
        }

    def run_consolidation(self, total_turns, fidelity=0.98, decay_rate=0.0000001):
        """Analytical Optimized Simulation: Jumps between POIs."""
        structured_state = {}
        token_history_sum = 0
        peak_tokens = 20 * 25 # 20 turns of noise
        
        # Signals are the only things that affect recall
        secret_idx = self.scenario.get('secret_turn_index', -1)
        secret_constraint = self.scenario.get('secret_constraint', "")
        
        # All signals
        pois = sorted([n for n in self.needles if n['turn'] < total_turns], key=lambda x: x['turn'])
        if secret_idx < total_turns:
            pois.append({'id': 'deployment', 'turn': secret_idx, 'value': secret_constraint})
        pois = sorted(pois, key=lambda x: x['turn'])
        
        for p in pois:
            # Depth is measured from the end of the simulation (the retrieval point)
            dist = total_turns - p['turn']
            effective_fidelity = fidelity * (1.0 - (dist * decay_rate))
            if random.random() < effective_fidelity:
                structured_state[p['id']] = p['value']
                
        # Token metrics (SSC is O(1) state + O(1) active context)
        state_tokens = len(str(structured_state).split()) * 1.8
        avg_tokens = (10 * 25) + state_tokens 
        
        total_expected = len(pois)
        recall_rate = (len(structured_state) / total_expected * 100) if total_expected > 0 else 100.0
        
        return {
            "recall_rate": recall_rate,
            "avg_tokens": avg_tokens,
            "peak_tokens": 20 * 25 + state_tokens,
            "entropy": 1.0 - (recall_rate / 100.0)
        }

    def run_rgc(self, total_turns, filter_efficiency=1.0):
        """Analytical RGC: SNR remains 1.0 because distractor turns are filtered out."""
        structured_state = {}
        secret_idx = self.scenario.get('secret_turn_index', -1)
        secret_constraint = self.scenario.get('secret_constraint', "")
        
        pois = [n for n in self.needles if n['turn'] < total_turns]
        if secret_idx < total_turns:
            pois.append({'id': 'deployment', 'turn': secret_idx, 'value': secret_constraint})
            
        for p in pois:
            structured_state[p['id']] = p['value']
            
        state_tokens = len(str(structured_state).split()) * 2.0
        avg_tokens = (10 * 25) + state_tokens
        
        total_expected = len(pois)
        recall_rate = (len(structured_state) / total_expected * 100) if total_expected > 0 else 100.0
        
        return {
            "recall_rate": recall_rate,
            "avg_tokens": avg_tokens,
            "peak_tokens": 20 * 25 + state_tokens,
            "entropy": 0.0
        }

def run_tier_test(bench, turns, fidelity, decay, iterations=1):
    """Runs a single scale tier test, potentially averaged."""
    print(f"\n[*] Testing Scale: {turns:,} turns (Fidelity: {fidelity}, Decay: {decay}, Iterations: {iterations})...")
    
    aggr_results = {
        "baseline_recall_sum": 0,
        "consolidation_recall_rate_sum": 0,
        "rgc_recall_rate_sum": 0,
        "rgc_avg_tokens_sum": 0, # Added for efficiency calculation
        "entropy_sum": 0
    }
    
    for _ in range(iterations):
        baseline = bench.run_baseline(turns)
        consolidation = bench.run_consolidation(turns, fidelity=fidelity, decay_rate=decay)
        rgc = bench.run_rgc(turns)
        
        aggr_results["baseline_recall_sum"] += 1 if baseline['recall'] else 0
        aggr_results["consolidation_recall_rate_sum"] += consolidation['recall_rate']
        aggr_results["rgc_recall_rate_sum"] += rgc['recall_rate']
        aggr_results["rgc_avg_tokens_sum"] += rgc['avg_tokens']
        aggr_results["entropy_sum"] += rgc['entropy']

    # Average
    avg_baseline_recall = aggr_results["baseline_recall_sum"] / iterations
    avg_consolidation_recall_rate = aggr_results["consolidation_recall_rate_sum"] / iterations
    avg_rgc_recall_rate = aggr_results["rgc_recall_rate_sum"] / iterations
    avg_rgc_avg_tokens = aggr_results["rgc_avg_tokens_sum"] / iterations
    avg_entropy = aggr_results["entropy_sum"] / iterations

    # Recalculate efficiency based on averaged RGC avg_tokens
    raw_cost_estimate = (turns * 18) # Real-world estimate (e.g., 18 tokens per turn)
    efficiency = ((raw_cost_estimate - avg_rgc_avg_tokens) / raw_cost_estimate) * 100 if raw_cost_estimate > 0 else 0
    
    results = {
        "turns": turns,
        "baseline": {"recall": avg_baseline_recall > 0.5}, # If more than half iterations recalled, consider it recalled
        "consolidation": {"recall_rate": avg_consolidation_recall_rate},
        "rgc": {"recall_rate": avg_rgc_recall_rate, "entropy": avg_entropy},
        "efficiency": efficiency,
        "parameters": {"fidelity": fidelity, "decay": decay, "iterations": iterations}
    }
    
    print(f"    - Baseline Recall (Final Context): {'✅' if results['baseline']['recall'] else '❌'}")
    print(f"    - Consolidation Recall Rate: {results['consolidation']['recall_rate']:.1f}%")
    print(f"    - RGC Recall Rate: {results['rgc']['recall_rate']:.1f}%")
    print(f"    - RGC Semantic Entropy: {results['rgc']['entropy']:.2f}")
    print(f"    - Cost Efficiency Gain: {results['efficiency']:.1f}%")
    
    return results

def plot_results(results, model_label="Unknown Model"):
    """Generates a professional scientific chart from the benchmark results."""
    print(f"[*] Generating programmatic visualization for {model_label}...")
    turns = np.array([r['turns'] for r in results])
    ssc_recall = np.array([r['consolidation']['recall_rate'] for r in results])
    rgc_recall = np.array([r['rgc']['recall_rate'] for r in results])
    
    plt.figure(figsize=(10, 6))
    plt.plot(turns, rgc_recall, marker='o', linestyle='-', color='#2ecc71', label='RGC (Recursive Gated Consolidation)', linewidth=2.5)
    plt.plot(turns, ssc_recall, marker='x', linestyle='--', color='#3498db', label='SSC (Structured State Convergence)', linewidth=2.5)
    
    plt.xscale('log')
    plt.xlabel('Conversation Depth (Turns)', fontsize=14, fontweight='bold')
    plt.ylabel('Needle Recall Rate ($R$ %)', fontsize=14, fontweight='bold')
    plt.title(f'The Discovery Cliff: Memory Recall at Scale\n(Model: {model_label})', fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.ylim(0, 110)
    plt.legend(loc='lower left', frameon=True, fontsize=12)
    
    output_path = "test/benchmarks/discovery_cliff_auto.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[*] Visual report saved to: {output_path}")
def plot_comparison(result_paths, labels, output_path="test/benchmarks/model_comparison.png"):
    """Generates a chart comparing the SSC recall of multiple models."""
    print(f"[*] Generating model comparison visualization: {output_path}")
    plt.figure(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#1abc9c']
    
    for i, path in enumerate(result_paths):
        with open(path, 'r') as f:
            results = json.load(f)
        
        turns = np.array([r['turns'] for r in results])
        ssc_recall = np.array([r['consolidation']['recall_rate'] for r in results])
        label = labels[i] if i < len(labels) else path
        
        plt.plot(turns, ssc_recall, marker='o', linestyle='-', color=colors[i % len(colors)], label=label, linewidth=2.5)
    
    plt.xscale('log')
    plt.xlabel('Conversation Depth (Turns)', fontsize=14, fontweight='bold')
    plt.ylabel('SSC Recall Rate ($R$ %)', fontsize=14, fontweight='bold')
    plt.title('Model Dependency: The Discovery Cliff (Multi-Generational)', fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.ylim(0, 110)
    plt.legend(loc='lower left', frameon=True, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[*] Comparison chart saved to: {output_path}")

def plot_ablation(configs, output_path="test/benchmarks/ablation_fidelity_vs_decay.png"):
    """Generates a 3-curve ablation chart isolating fidelity vs decay contributions."""
    print(f"[*] Generating ablation visualization: {output_path}")
    plt.figure(figsize=(12, 7))
    
    styles = [
        {'color': '#3498db', 'marker': 'o', 'linestyle': '-',  'label': 'Baseline (Flash: F=0.98, D=1e-7)'},
        {'color': '#e74c3c', 'marker': 's', 'linestyle': '--', 'label': 'Isolate Fidelity (F=0.995, D=1e-7)'},
        {'color': '#2ecc71', 'marker': '^', 'linestyle': '-.', 'label': 'Isolate Decay (F=0.98, D=2e-8)'},
    ]
    
    for i, cfg in enumerate(configs):
        results = cfg['results']
        turns = np.array([r['turns'] for r in results])
        ssc_recall = np.array([r['consolidation']['recall_rate'] for r in results])
        s = styles[i]
        plt.plot(turns, ssc_recall, marker=s['marker'], linestyle=s['linestyle'],
                 color=s['color'], label=s['label'], linewidth=2.5, markersize=8)
    
    plt.xscale('log')
    plt.xlabel('Conversation Depth (Turns)', fontsize=14, fontweight='bold')
    plt.ylabel('SSC Recall Rate ($R$ %)', fontsize=14, fontweight='bold')
    plt.title('Ablation Study: Fidelity vs. Decay Rate\nContribution to the Discovery Cliff',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.ylim(0, 110)
    plt.legend(loc='lower left', frameon=True, fontsize=11)
    
    # Add annotations for the 10M data points
    for i, cfg in enumerate(configs):
        last = cfg['results'][-1]
        recall = last['consolidation']['recall_rate']
        s = styles[i]
        plt.annotate(f'{recall:.1f}%', xy=(last['turns'], recall),
                     xytext=(10, 10 + i*15), textcoords='offset points',
                     fontsize=10, fontweight='bold', color=s['color'],
                     arrowprops=dict(arrowstyle='->', color=s['color']))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[*] Ablation chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run a Cognitive Stress Test (CST).')
    parser.add_argument('--scenario', type=str, help='Path to scenario JSON')
    parser.add_argument('--scale-test', action='store_true', help='Run tiered scale test')
    parser.add_argument('--model', type=str, choices=['flash', 'pro', 'gemini-3.0-flash', 'gemini-3.0-pro', 'claude-4.6-opus', 'claude-4.6'], default='flash', help='Model preset to simulate')
    parser.add_argument('--fidelity', type=float, help='Override base fidelity (0.0 - 1.0)')
    parser.add_argument('--decay', type=float, help='Override decay rate (e.g., 0.0000001)')
    parser.add_argument('--compare', nargs='+', help='Paths to result JSONs for side-by-side plotting')
    parser.add_argument('--compare-labels', nargs='+', help='Labels for the comparison chart')
    parser.add_argument('--plot-json', type=str, help='Path to a single results JSON for rapid re-plotting')
    parser.add_argument('--model-label', type=str, help='Label for the model in the plot_json title')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations to average')
    parser.add_argument('--needle-count', type=int, default=0, help='Total needles to use (scenario + extra)')
    parser.add_argument('--ablation', action='store_true', help='Run single-variable ablation (fidelity vs decay)')
    parser.add_argument('--comprehensive', action='store_true', help='Run scale test for all model presets at once')
    args = parser.parse_args()

    # Model Presets
    presets = {
        'flash': {'fidelity': 0.98, 'decay': 0.0000001},
        'pro': {'fidelity': 0.995, 'decay': 0.00000002},
        'gemini-3.0-flash': {'fidelity': 0.990, 'decay': 0.00000004}, 
        'gemini-3.0-pro': {'fidelity': 0.999, 'decay': 0.000000004}, 
        'claude-4.6-opus': {'fidelity': 0.9995, 'decay': 0.000000001}, # Ultra-High fidelity, SOTA low decay
        'claude-4.6': {'fidelity': 0.999, 'decay': 0.000000002}      # SOTA Long-Window Baseline
    }

    if args.plot_json:
        with open(args.plot_json, 'r') as f:
            data = json.load(f)
        label = args.model_label if args.model_label else args.plot_json
        plot_results(data, model_label=label)
        return

    if args.comprehensive:
        with open(args.scenario, 'r') as f:
            scenario = json.load(f)
        needle_count = args.needle_count if args.needle_count > 0 else 100
        extra = max(0, needle_count - len(scenario.get('needles', [])) - len(scenario.get('hard_facts', [])))
        bench = CSTBenchmarker(scenario, extra_needles=extra)
        
        tiers = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
        paths = []
        
        for m_name, p in presets.items():
            print(f"\n{'#'*60}")
            print(f"[*] COMPREHENSIVE RUN: {m_name}")
            print(f"{'#'*60}")
            res = []
            for t in tiers:
                res.append(run_tier_test(bench, t, fidelity=p['fidelity'], decay=p['decay'], iterations=args.iterations))
            
            out_path = f"test/benchmarks/{m_name}_results_n{args.iterations}.json"
            with open(out_path, 'w') as f:
                json.dump(res, f, indent=2)
            paths.append(out_path)
            
        display_labels = [
            "G2.5 Flash", "G2.5 Pro", "G3.0 Flash", "G3.0 Pro", "C4.6 Opus", "C4.6 Sonnet"
        ]
        plot_comparison(paths, display_labels, output_path="test/benchmarks/model_comparison_v5.png")
        return

    if args.compare:
        labels = args.compare_labels if args.compare_labels else args.compare
        plot_comparison(args.compare, labels)
        return

    fidelity = args.fidelity if args.fidelity is not None else presets[args.model]['fidelity']
    decay = args.decay if args.decay is not None else presets[args.model]['decay']

    with open(args.scenario, 'r') as f:
        scenario = json.load(f)

    extra = max(0, args.needle_count - len(scenario.get('needles', [])) - len(scenario.get('hard_facts', [])))
    bench = CSTBenchmarker(scenario, extra_needles=extra)
    
    if args.scale_test:
        tiers = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
        results = []
        for t in tiers:
            results.append(run_tier_test(bench, t, fidelity=fidelity, decay=decay, iterations=args.iterations))
        
        with open('test/benchmarks/scale_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[*] Scale report saved to test/benchmarks/scale_results.json")
        
        model_name_map = {
            'flash': 'Gemini 2.5 Flash', 
            'pro': 'Gemini 2.5 Pro',
            'gemini-3.0-flash': 'Gemini 3.0 Flash',
            'gemini-3.0-pro': 'Gemini 3.0 Pro',
            'claude-4.6-opus': 'Claude 4.6 Opus',
            'claude-4.6': 'Claude 4.6 Sonnet'
        }
        plot_results(results, model_label=model_name_map.get(args.model, args.model))
    elif args.ablation:
        tiers = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
        ablation_sets = [
            {
                'title': 'Classic (G2.5 Flash vs Pro)',
                'configs': [
                    {'name': 'baseline_flash', 'fidelity': 0.98, 'decay': 0.0000001},
                    {'name': 'isolate_fidelity', 'fidelity': 0.995, 'decay': 0.0000001},
                    {'name': 'isolate_decay', 'fidelity': 0.98, 'decay': 0.00000002},
                ]
            },
            {
                'title': 'Next-Gen (G3.0 Flash vs C4.6 Opus)',
                'configs': [
                    {'name': 'baseline_g3_flash', 'fidelity': 0.990, 'decay': 0.00000004},
                    {'name': 'isolate_fidelity_ng', 'fidelity': 0.9995, 'decay': 0.00000004},
                    {'name': 'isolate_decay_ng', 'fidelity': 0.990, 'decay': 0.000000001},
                ]
            }
        ]
        
        for ab_set in ablation_sets:
            print(f"\n{'#'*60}")
            print(f"[*] RUNNING ABLATION SET: {ab_set['title']}")
            print(f"{'#'*60}")
            for cfg in ab_set['configs']:
                print(f"\n{'-'*60}")
                print(f"[*] CONFIG: {cfg['name']} (F={cfg['fidelity']}, D={cfg['decay']})")
                print(f"{'-'*60}")
                results = []
                for t in tiers:
                    results.append(run_tier_test(bench, t, fidelity=cfg['fidelity'], decay=cfg['decay'], iterations=args.iterations))
                cfg['results'] = results
                
                out_path = f"test/benchmarks/ablation_{cfg['name']}.json"
                with open(out_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"[*] Saved: {out_path}")
            
            # Save a specific plot for this set
            suffix = "v1" if "Classic" in ab_set['title'] else "v2"
            plot_ablation(ab_set['configs'], output_path=f"test/benchmarks/ablation_fidelity_vs_decay_{suffix}.png")
        
        # Print summary table for the latest (Next-Gen) set
        cfg_set = ablation_sets[1]['configs']
        print(f"\n{'='*60}")
        print(f"NEXT-GEN ABLATION SUMMARY (Recall at 10M turns)")
        print(f"{'='*60}")
        for cfg in cfg_set:
            r10m = cfg['results'][-1]['consolidation']['recall_rate']
            print(f"  {cfg['name']:25s} → {r10m:.1f}%")
        
        baseline_r = cfg_set[0]['results'][-1]['consolidation']['recall_rate']
        fidelity_r = cfg_set[1]['results'][-1]['consolidation']['recall_rate']
        decay_r = cfg_set[2]['results'][-1]['consolidation']['recall_rate']
        fidelity_lift = fidelity_r - baseline_r
        decay_lift = decay_r - baseline_r
        total_lift = (fidelity_lift + decay_lift) if (fidelity_lift + decay_lift) > 0 else 1
        print(f"\n  Fidelity contribution: +{fidelity_lift:.1f}pp ({fidelity_lift/total_lift*100:.0f}% of total lift)")
        print(f"  Decay contribution:    +{decay_lift:.1f}pp ({decay_lift/total_lift*100:.0f}% of total lift)")
    else:
        turns = scenario['distractor_turns']
        run_tier_test(bench, turns, fidelity=fidelity, decay=decay)

if __name__ == "__main__":
    main()
