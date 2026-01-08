# Fairness-Aware Adaptive Bitrate Streaming on Edge Layers

This repository contains the experimental code and infrastructure used in the Ph.D. thesis:

**“Fairness-Aware Adaptive Bitrate Streaming on Edge Layers”**  
Author: André Luiz Silva de Moraes

The repository provides a reproducible implementation of a fairness-aware adaptive bitrate (ABR) framework evaluated in edge computing scenarios, focusing on multi-client adaptive streaming and fairness assessment.

---

## Repository Structure

├── experiments/
│ └── experiment-5-client-aware-fairness-healthcare/
│ ├── abr_quality_selector.py
│ ├── fairness_utils.py
│ ├── metric_processing.py
│ ├── decision_module.py
│ ├── hotdash.py
│ ├── run_clients_in_docker.py
│ ├── simulations/
│ ├── traces/
│ └── tests/
│
├── infrastructure/
│ └── 06-origin-2-edge-clients/
│ ├── docker-compose.yml
│ ├── DockerfileClient
│ ├── DockerfileServer
│ ├── conf.d/
│ ├── html-origin/
│ └── html-edge/
│
├── LICENSE
└── README.md



---

## Experimental Scope

This repository includes:
- The final client-aware fairness experiment adopted in the thesis
- Adaptive bitrate (ABR) policies, including heuristic, buffer-based, and reinforcement learning–based approaches
- Fairness metrics implementation (e.g., TF-QoE, QFS, BFI)
- Docker-based infrastructure for origin and edge servers
- Scripts to reproduce the experimental execution and metric collection

---

## Data and Multimedia Assets

- Video content is based on **Big Buck Bunny**, which is publicly available and licensed for research use.
- Network traces are derived from publicly available datasets (e.g., Mahimahi-compatible traces).
- To avoid unnecessary data duplication, large video segments and generated datasets are not stored directly in this repository.
- Instructions and configuration files required to regenerate the multimedia pipeline are provided.

---

## Reproducibility Notes

The repository is designed to support reproducibility of the experimental setup described in the thesis.  
Due to storage and licensing considerations:
- Raw logs, large CSV result files, and intermediate trained RL models are intentionally excluded.
- The provided code and configuration files are sufficient to reproduce the experimental workflow.

---

## License

This project is released under the **MIT License**.

