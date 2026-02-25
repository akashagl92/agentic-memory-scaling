# Agentic Memory Scaling

**Scaling Laws for Memory Consolidation in LLM-Based Agentic Systems**

[![Paper](https://img.shields.io/badge/Paper-APA_Format-blue)](paper/paper_memory_consolidation_apa.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Abstract

This repository contains the research paper, benchmark data, and reproducibility harness for **"The Discovery Cliff: Scaling Laws for Memory Consolidation in LLM-Based Agentic Systems."**

We identify a fundamental scaling limit—the **Discovery Cliff**—where standard single-stage memory consolidation (SSC) fails to extract new signals from long conversation histories. Through a dual-tier ablation study across multiple model generations (Google Gemini 2.5/3.0, Anthropic Claude 4.6), we prove that **temporal decay accounts for 99% of the recall collapse**, establishing an invariant Scaling Law for Agentic Memory.

We propose **Recursive Gated Consolidation (RGC)**, a two-stage architecture that maintains **100% signal recall at 10M+ turns** by decoupling discovery from synthesis.

## Key Findings

| Finding | Detail |
| :--- | :--- |
| **The Discovery Cliff** | SSC recall collapses after 1M turns (Flash) / 5M turns (Pro) |
| **Temporal Decay Dominance** | Decay rate ($d$) accounts for **99%** of recall collapse |
| **RGC Performance** | Maintains **100% recall** at 10M+ turns |
| **Synthetic Saturation** | Code distractors accelerate decay by ~12% vs. prose |
| **Cross-System Validation** | Validated in chatbot (Moltbot) and IDE (Antigravity) environments |

## Repository Structure

```
agentic-memory-scaling/
├── README.md                  # This file
├── LICENSE                    # MIT License
├── paper/
│   └── paper_memory_consolidation_apa.md  # Full APA research paper
├── benchmarks/
│   ├── results/               # High-fidelity JSON results (N=50)
│   │   ├── flash_results_n50.json
│   │   ├── pro_results_n50.json
│   │   ├── gemini-3.0-flash_results_n50.json
│   │   ├── claude-4.6-opus_results_n50.json
│   │   └── ablation_*.json
│   └── figures/               # Publication-ready figures (v5)
│       ├── discovery_cliff_v5.png
│       ├── model_comparison_v5.png
│       ├── ablation_fidelity_vs_decay_v1.png
│       └── ablation_fidelity_vs_decay_v2.png
└── scripts/
    └── run_cst.py             # Cognitive Stress Test harness
```

## Reproducing Results

### Prerequisites
- Python 3.10+
- `matplotlib` (for visualization)

### Running the Benchmark
```bash
cd scripts
python run_cst.py
```

This will execute the Cognitive Stress Test (CST) harness, generating:
- `scale_results.json` — Raw recall data across logarithmic turn depths
- `discovery_cliff_auto.png` — The Discovery Cliff visualization

## Citation

If you use this work in your research, please cite:

```bibtex
@article{agrawal2026discovery,
  title={The Discovery Cliff: Scaling Laws for Memory Consolidation in LLM-Based Agentic Systems},
  author={Agrawal, Akash},
  year={2026},
  note={Preprint}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
