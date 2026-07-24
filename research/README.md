# Research & Legacy Experimental Code

This directory contains optional research extensions, legacy experimental models, and exploratory tools created during early development.

## Structure

```
research/
├── README.md
└── legacy_student_t_hmm/
    ├── README.md
    ├── regime_detector.py      # Custom Student-t HMM with Baum-Welch & Viterbi
    ├── regime_features.py      # 54-dimensional feature extraction pipeline
    ├── transaction_costs.py    # Almgren-Chriss market impact cost model
    ├── monte_carlo.py          # Bootstrap & stress testing tools
    ├── stats.py                # Specialized statistical routines
    └── visualize.py            # Comprehensive research plotting module
```

## Important Non-Import Policy

> [!WARNING]
> None of the modules in this directory are imported by or linked into the official `regime_shift` package (`src/regime_shift`).
>
> They are preserved strictly for academic reference and future research comparison. The official submission foundation uses standard, audited packages (such as `hmmlearn` and `cvxpy`) to guarantee reproducibility, robustness, and zero look-ahead bias.
