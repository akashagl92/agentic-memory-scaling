# Agentic Memory Scaling: Structured State Convergence and the Discovery Cliff

**Scaling Laws for Memory Consolidation in LLM-Based Agentic Systems at $10^7$ Turns**

[![Paper](https://img.shields.io/badge/Paper-APA_Format-blue)](paper/paper_memory_consolidation_apa.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Abstract

This repository contains the research paper, benchmark data, and reproducibility harness for **"Structured State Convergence and the Discovery Cliff: O(1) Memory Scaling for LLM Agents at $10^7$ Turns."**

We identify a fundamental scaling limit—the **Discovery Cliff**—where standard single-stage memory consolidation (SSC) fails to extract new signals from long conversation histories. Using the constants derived from Live API runs (Tier 1), we perform $10^7$-turn Monte Carlo simulations $(N=1000)$ across multiple model generations (Google Gemini 2.5/3.0/3.1, Anthropic Claude 4.6). Results demonstrate that **temporal decay accounts for up to 99% of the recall collapse** in next-generation models, establishing an invariant Scaling Law for Agentic Memory. At extreme scale ($10^7$ turns), SSC recall degrades to **16.8%**.

We evaluate **Recursive Gated Consolidation (RGC)**, a two-stage architecture that eliminates this decay and maintains **100% signal recall at scale** by decoupling discovery from history depth. This repository contains the RGC Benchmark Suite, empirical calibration scripts, and the formal publication detailing these findings.

## Key Findings

| Finding                      | Detail                                                         |
| :--------------------------- | :------------------------------------------------------------- |
| **The Discovery Cliff**      | SSC recall collapses to **17.0%** at 10M turns (Flash)         |
| **Temporal Decay Dominance** | Decay rate ($d$) accounts for up to **99%** of recall collapse |
| **RGC Performance**          | Maintains **100% recall** at 10M+ turns                        |
| **Inverted Latency Law**     | Latency stabilizes/drops as context grows (Gemini 3.0)         |
| **Hardware Grounding**       | Validated via TPU v4 OCS and SparseCore optimizations          |
| **Cross-System Validation**  | Validated in chatbot (Moltbot) and IDE (Antigravity)           |

## Repository Structure

```
agentic-memory-scaling/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── paper/
│   └── paper_memory_consolidation_apa.md  # Full APA research paper
├── benchmarks/
│   ├── results/               # High-fidelity JSON results (N=1000)
│   │   ├── flash_results_n1000.json
│   │   ├── pro_results_n1000.json
│   │   ├── gemini-3.0-flash_results_n1000.json
│   │   ├── claude-4.6-opus_results_n1000.json
│   │   └── ablation_*.json
│   └── figures/               # Publication-ready figures (v6)
│       ├── discovery_cliff_auto.png
│       ├── model_comparison_v6_final.png
│       ├── ablation_fidelity_vs_decay_v1.png
│       └── ablation_fidelity_vs_decay_v2.png
└── scripts/
    ├── run_cst.py             # Cognitive Stress Test harness
    └── lib_diag.py            # Live API diagnostic implementation
```

## Scaling Laws & Hardware Grounding

We introduce the **Inverted Latency Scaling Law**, observing that on Gemini 3.0 Flash, mean latency triggers a sharp transition from **15.33s (5k turns)** down to a deterministic **2.08s (10k turns)**.

This stabilization is supported by modern infrastructure designs such as Google's **TPU v4/v5** pods. By leveraging **Optical Circuit Switches (OCSes)** for millisecond-level topology reconfiguration and **SparseCores** for embedding acceleration (Jouppi et al., 2023), RGC-enabled agents move from unoptimized $O(n)$ tensor reads to highly localized, hardware-optimized pathways.

## Reproducing Results

### Prerequisites

- Python 3.10+
- `matplotlib` (for visualization)

### Agentic Memory Scaling: Structured State Convergence and the Discovery Cliff

> [!NOTE]
> This repository contains the data, figures, and research paper for the study of $O(1)$ memory scaling at $10^7$ turns.

- `scale_results.json` — Raw recall data across logarithmic turn depths
- `discovery_cliff_auto.png` — The Discovery Cliff visualization

## Citation

If you use this work in your research, please cite:

```bibtex
@article{agrawal2026discovery,
  title={Structured State Convergence for O(1) Memory Scaling in LLM Agents: An Empirical Study of Discovery Limits at 10 Million Turns},
  author={Agrawal, Akash},
  year={2026},
  note={Preprint}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
