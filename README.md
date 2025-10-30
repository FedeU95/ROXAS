# ROXAS
ROXAS (Reasoning Over eXplained AnomalieS) is a methodology integrating explainable AI (XAI) and large language model (LLM) reasoning to move from model interpretation to actionable mitigation in network defense.

# OVERVIEW

Next-generation network infrastructures introduce unprecedented complexity and a rapidly expanding attack surface.
While AI-based anomaly detection models can identify unknown threats, their lack of interpretability often hinders effective response.

ROXAS operationalizes a multi-stage defense methodology that:

Detects anomalies using an XGBoost regressor trained on benign data,

Explains each alert through logic-based feature attribution, and

Reasons over these explanations using LLM to generate mitigation guidance.

The framework’s reasoning outputs are evaluated against the MITRE FIGHT database, demonstrating alignment with expert best practices in 5G network defense.

# CITATION

if you use this repo, please refer to:

[To be updated upon publication]


# ACK
This work was carried out within the NEST project AIR2, which is
partially supported by the Wallenberg AI, Autonomous Systems and Software
Program (WASP) funded by the Knut and Alice Wallenberg Foundation. It
was also supported by ELLIIT, the Excellence Center at Link¨oping – Lund in
Information Technology, and the US NSF under grants 2112471 and 2229876.