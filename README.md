# ROXAS

**ROXAS (Reasoning Over eXplained AnomalieS)** is a methodology integrating explainable artificial intelligence (XAI) and large language model (LLM) to move from anomaly detection and explanation toward actionable mitigation in 5G network defense.

## Overview

AI-based anomaly detection can identify previously unseen or abnormal network behavior, but detecting an anomaly alone provides limited support for deciding how to respond to it. ROXAS addresses this gap through a multi-stage methodology that:

1. Detect anomalies using an XGBoost regressor trained on benign network behavior;
2. Interpret detected anomalies through logic-based feature attribution; and
3. Process these explanations using an LLM to generate mitigation recommendations.

The original ROXAS methodology was introduced and evaluated using 5G Non-Access Stratum (NAS) traces. The extended version of ROXAS expands the evaluation to both **Non-Access Stratum (NAS)** and **Radio Resource Control (RRC)** traces, allowing us to investigate how protocol-layer characteristics affect anomaly detection and feature attribution.

For mitigation generation, the extended methodology considers two approaches:

- **Unconstrained generation**, where the LLM generates mitigation recommendations directly from the anomaly explanations;
- **MITRE FiGHT-grounded generation**, where mitigation generation is constrained using domain knowledge from the MITRE FiGHT framework.

The resulting recommendations are evaluated through both **human expert assessment** and **LLM-assisted assessment**. This allows us to examine the relationship between attack-specific mitigation guidance and alignment with an established 5G security taxonomy, as well as the reliability of LLM-assisted evaluation.

The `archive/` directory contains experimental material from the original ROXAS study and is retained for reproducibility and reference. The current experimental pipeline is contained in the main repository structure.

## Dataset

The NAS dataset (`msa_nas.csv`) used in the original ROXAS evaluation is publicly available on Zenodo:

https://zenodo.org/records/14720824

Additional RRC traces were collected in the extended evaluation.

# MITRE FiGHT Data

ROXAS uses mitigation information from the MITRE FiGHT knowledge base to
ground constrained LLM mitigation generation.

This repository does not redistribute the complete MITRE FiGHT database.
It contains only the processed information required by the ROXAS experiments.

For the complete and current MITRE FiGHT knowledge base, please refer to
the official MITRE FiGHT project.

The scripts in this directory document the processing used to obtain the
mitigation information consumed by the ROXAS pipeline.

## Citation

If you use ROXAS or material from this repository, please cite the original ROXAS publication:

> F. Uccello, I. Karim, K. S. Mubasshir, E. Bertino, and S. Nadjm-Tehrani,  
> “Mission Explainable: From Feature Attribution to Mitigation in 5G Anomaly Detection,”  
> *2026 International Conference on Computing, Networking and Communications (ICNC)*,  
> pp. 1–6, 2026.  
> DOI: 10.1109/ICNC68183.2026.11416967

### BibTeX

```bibtex
@inproceedings{uccello2026mission,
  author    = {Federica Uccello and
               Imtiaz Karim and
               Kazi Samin Mubasshir and
               Elisa Bertino and
               Simin Nadjm-Tehrani},
  title     = {Mission Explainable: From Feature Attribution to Mitigation
               in 5G Anomaly Detection},
  booktitle = {2026 International Conference on Computing, Networking
               and Communications (ICNC)},
  pages     = {1--6},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/ICNC68183.2026.11416967}
}
```

The citation information will be updated upon publication of the extended version of ROXAS.

## Related Publications

ROXAS was originally introduced in the ICNC 2026 paper above as a feasibility study investigating whether LLMs can translate feature-level anomaly explanations into operational mitigation guidance.

The extended work builds on this methodology in three main directions:

1. comparison of anomaly detection and feature attribution across the NAS and RRC protocol layers;
2. comparison of unconstrained and MITRE FiGHT-grounded mitigation generation; and
3. systematic validation of mitigation mappings using human experts and LLM-assisted assessment.

## Acknowledgments

This work was carried out within the **NEST project AIR2**, which is partially supported by the **Wallenberg AI, Autonomous Systems and Software Program (WASP)**, funded by the Knut and Alice Wallenberg Foundation.

It was also supported by **ELLIIT, the Excellence Center at Linköping–Lund in Information Technology**, and the **U.S. National Science Foundation (NSF)** under grants **2112471** and **2229876**.