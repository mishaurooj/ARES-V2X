# ARES-V2X

## Adaptive Resilient Edge-Cloud Security for Trust-Aware and Byzantine-Resilient V2X Cyber Defense

---

# Project Overview

ARES-V2X is an agentic cyber-resilience framework designed for secure Vehicle-to-Everything (V2X) communication environments. The framework combines:

* edge intelligence
* cloud-assisted orchestration
* trust-aware reasoning
* graph-based anomaly propagation
* adaptive consensus fusion
* temporal resilience modeling
* Byzantine-resilient orchestration
* LLM-assisted policy reasoning
* self-healing reliability adaptation

The project evolved from a standard multiclass V2X intrusion detection pipeline into a full cyber-resilience orchestration framework aligned with IEEE Transactions on Dependable and Secure Computing style research.

Unlike traditional V2X IDS pipelines that focus only on classification accuracy, ARES-V2X focuses on:

* operational resilience
* adaptive recovery
* survivability under corruption
* distributed trust management
* dependable edge-cloud coordination
* explainable response orchestration

The framework uses VeReMi-NextGen multiclass V2X attack datasets and evaluates resilience under multiple realistic failure conditions.

---

# Research Motivation

Modern V2X environments operate under highly dynamic and unreliable conditions. Vehicles continuously exchange safety-critical messages involving:

* vehicle position
* speed
* acceleration
* heading
* braking state
* congestion information
* road awareness

Traditional centralized intrusion detection systems fail in these environments because:

* communication latency fluctuates
* edge nodes become unreliable
* trust relationships evolve dynamically
* attackers manipulate distributed consensus
* graph relationships become corrupted
* temporal behavior drifts over time
* cloud connectivity is unstable

Most existing V2X intrusion detection studies focus only on static classification performance using:

* RandomForest
* XGBoost
* CNN
* LSTM
* GNN

However, real V2X deployments require:

* adaptive orchestration
* resilient consensus
* trust-aware reasoning
* operational recovery
* survivability under Byzantine corruption
* edge deployment feasibility

ARES-V2X was designed to address these missing operational requirements.

---

# Dataset Overview

The project uses the VeReMi-NextGen dataset.

The dataset contains multiple realistic V2X attack scenarios across:

* highway environments
* urban environments
* low-density traffic
* high-density traffic

Attack categories include:

| Attack Type            | Description                    |
| ---------------------- | ------------------------------ |
| constantPositionOffset | fixed GPS manipulation         |
| randomPositionOffset   | randomized location corruption |
| trafficCongestionSybil | fake congestion generation     |
| suddenStop             | false emergency stop injection |
| reversedHeading        | manipulated vehicle heading    |
| timeDelayAttack        | delayed message forwarding     |
| dosAttack              | denial-of-service flooding     |
| dataReplay             | replayed safety messages       |
| feignedBraking         | false braking behavior         |
| zeroSpeedReport        | fake stationary reports        |

The dataset provides:

* multiclass labels
* binary labels
* train/validation/test partitions
* multiple scenario distributions
* edge-driving behavior
* mobility variations

---

# Evolution of the Project

ARES-V2X evolved through multiple major redesign stages.

Each stage solved a critical limitation.

---

# Stage 1: Initial Multiclass Detection Pipeline

## Objective

Build a multiclass V2X intrusion detection system.

## Initial Approach

The first implementation focused only on:

* RandomForest
* XGBoost
* ExtraTrees
* standard feature engineering

The system used:

* sender position
* speed
* acceleration
* heading
* delay
* distance features

## Problems

The initial design suffered from:

* poor operational realism
* no resilience modeling
* no temporal adaptation
* no trust propagation
* no explainability
* no deployment analysis
* no fault tolerance

The framework looked like a standard ML benchmark paper.

This was insufficient for IEEE TDSC.

---

# Stage 2: Temporal Trust Modeling

## Objective

Add dynamic behavioral reasoning.

## Added Components

The framework introduced:

* rolling temporal windows
* temporal instability scoring
* trust decay estimation
* sender behavior history
* edge violation history
* delay evolution analysis

## Key Features Added

| Feature                  | Purpose                      |
| ------------------------ | ---------------------------- |
| sender_spd_roll_mean_5   | temporal speed stability     |
| sender_acl_roll_mean_5   | acceleration evolution       |
| sender_hed_diff          | heading drift                |
| temporal_instability     | dynamic behavior volatility  |
| sender_attack_rate_prior | historical attacker tendency |
| trust_decay              | trust degradation modeling   |

## What It Solved

This stage improved:

* temporal consistency
* adaptive behavioral profiling
* drift awareness
* historical anomaly reasoning

The system became more resilient to:

* replay attacks
* delayed attacks
* evolving attacker behavior

---

# Stage 3: Graph Trust Propagation

## Objective

Move from isolated message analysis to distributed relationship reasoning.

## Motivation

Vehicles do not operate independently.

A malicious sender affects:

* neighboring vehicles
* propagated trust relationships
* consensus reliability
* distributed traffic awareness

## Added Components

ARES-V2X introduced:

* graph neighbor aggregation
* propagated trust estimation
* local disagreement modeling
* graph risk diffusion
* neighborhood anomaly propagation

## Key Graph Features

| Feature                   | Purpose                         |
| ------------------------- | ------------------------------- |
| graph_neighbor_risk_prior | neighborhood risk propagation   |
| graph_trust_propagated    | distributed trust estimation    |
| graph_local_disagreement  | local behavioral inconsistency  |
| gnn_feature_disagreement  | neighborhood feature divergence |
| graph_sender_degree_prior | sender communication density    |

## Impact

This stage transformed the framework from:

"single-message detection"

to:

"distributed cyber-resilience reasoning"

This became one of the strongest novelty components.

---

# Stage 4: Adaptive Consensus Fusion

## Objective

Build dependable distributed decision making.

## Problem

Traditional ensemble voting assumes:

* all agents remain reliable
* all detectors contribute equally
* corruption does not occur

This assumption is unrealistic in V2X environments.

## Solution

ARES-V2X introduced:

* adaptive agent weighting
* reliability-aware fusion
* disagreement penalties
* self-healing weight adaptation
* operational resilience scoring

## Agents Used

| Agent           | Role                                |
| --------------- | ----------------------------------- |
| edge_agent      | low-latency local detection         |
| temporal_agent  | drift-aware reasoning               |
| trust_agent     | trust consistency estimation        |
| graph_agent     | neighborhood resilience analysis    |
| cloud_agent     | global context reasoning            |
| ml_fusion_agent | probabilistic classification fusion |

## Consensus Formula

ARES-V2X computes:

* weighted consensus risk
* disagreement penalties
* adaptive trust adjustment

This enabled:

* resilience-aware orchestration
* dependable consensus
* operational survivability

---

# Stage 5: Byzantine-Resilient Evaluation

## Objective

Evaluate survivability under corrupted internal agents.

## Why This Matters

Many V2X studies assume internal system components remain trustworthy.

This assumption fails in:

* compromised RSUs
* corrupted edge nodes
* manipulated trust systems
* poisoned graph propagation

## Byzantine Experiments Added

ARES-V2X evaluated:

| Corruption Scenario    | Description                        |
| ---------------------- | ---------------------------------- |
| trust_agent corruption | manipulated trust estimates        |
| graph_agent corruption | poisoned graph reasoning           |
| cloud_agent corruption | compromised cloud intelligence     |
| trust_graph corruption | simultaneous graph-trust poisoning |
| multi_byzantine        | coordinated corruption             |
| random_byzantine       | stochastic adversarial corruption  |

## Findings

ARES-V2X maintained reasonable survivability under:

* isolated corruption
* moderate distributed corruption

However:

* coordinated multi-agent corruption caused major degradation

This result was intentionally retained because:

* realistic limitations improve research credibility
* exaggerated robustness claims reduce trustworthiness

---

# Stage 6: Temporal Drift Recovery

## Objective

Evaluate long-term adaptation under evolving attacker behavior.

## Problem

Static intrusion detection models fail when:

* attacker strategies evolve
* communication patterns drift
* trust relationships change
* temporal distributions shift

## Added Components

ARES-V2X introduced:

* adaptive reliability updates
* self-healing weight evolution
* temporal drift simulations
* dynamic agent rebalancing

## Key Metrics

| Metric                    | Purpose                     |
| ------------------------- | --------------------------- |
| consensus_stability_score | fusion consistency          |
| agent_reliability_drift   | weight adaptation magnitude |
| resilience_utility        | operational survivability   |
| temporal recovery curves  | drift resilience            |

## Importance

This stage introduced true:

* adaptive resilience
* operational recovery
* agentic reliability evolution

This became a core TDSC contribution.

---

# Stage 7: LLM-Assisted Orchestration

## Objective

Introduce explainable operational response reasoning.

## Initial Problem

The early LLM layer only generated:

* textual summaries
* weak explanations

This was insufficient for:

* agentic AI claims
* operational orchestration
* explainable cyber defense

## Final Design

ARES-V2X transformed the LLM layer into:

* policy orchestration engine
* trust-aware response generator
* adaptive triage assistant
* recovery recommendation module

## LLM Tasks

The LLM generates:

| Capability                | Description                       |
| ------------------------- | --------------------------------- |
| incident triage           | operational severity analysis     |
| response recommendation   | isolation or monitoring decisions |
| adaptive policy reasoning | consensus threshold adaptation    |
| recovery planning         | trust restoration guidance        |
| explainability            | operational justification         |

## Ollama Integration

The project supports:

* local LLM inference
* llama3.2:3b
* offline orchestration reasoning

## Why This Matters

The LLM does not replace detection.

Instead, it performs:

* operational orchestration
* explainable decision support
* resilience-aware triage

This is the correct framing for publication.

---

# Stage 8: Edge-Cloud Deployment Analysis

## Objective

Evaluate practical deployment feasibility.

## Added Metrics

ARES-V2X introduced:

| Metric                      | Purpose                |
| --------------------------- | ---------------------- |
| latency_ms_per_msg          | inference efficiency   |
| estimated_memory_mb         | edge feasibility       |
| cloud_offload_ratio         | distributed processing |
| comm_overhead_bytes_per_msg | communication overhead |
| edge_cpu_percent            | deployment realism     |

## Motivation

Many V2X papers ignore:

* edge hardware constraints
* RSU limitations
* communication overhead
* deployment realism

ARES-V2X explicitly evaluates these factors.

This significantly strengthened the paper.

---

# Final System Architecture

The final ARES-V2X framework operates as:

```text
Raw V2X Messages
        ↓
Edge Feature Extraction
        ↓
Temporal Behavioral Modeling
        ↓
Trust Propagation Engine
        ↓
Graph Disagreement Analysis
        ↓
Adaptive Consensus Fusion
        ↓
Self-Healing Reliability Adaptation
        ↓
LLM Policy Orchestration
        ↓
Edge-Cloud Response Decision
```

---

# Python Files Overview

# 1. Dataset Processing Files

## Purpose

Convert VeReMi-NextGen JSON datasets into ML-ready CSV datasets.

## Key Operations

* JSON parsing
* attack labeling
* train/validation/test separation
* feature extraction
* class balancing

## Outputs

* balanced multiclass CSV
* engineered datasets
* merged scenario datasets

---

# 2. Temporal and Trust Engineering Files

## Purpose

Generate behavioral resilience features.

## Added Capabilities

* rolling temporal statistics
* drift estimation
* trust decay modeling
* historical attacker profiling

## Impact

Improved:

* temporal resilience
* adaptive reasoning
* long-term attacker awareness

---

# 3. Graph Trust Files

## Purpose

Build neighborhood-aware resilience reasoning.

## Added Features

* propagated trust
* graph disagreement
* neighborhood anomaly modeling
* graph feature divergence

## Impact

Enabled:

* distributed resilience
* graph-aware orchestration
* relationship-driven anomaly reasoning

---

# 4. Adaptive Consensus Files

## Purpose

Implement dependable distributed fusion.

## Added Features

* adaptive weighting
* disagreement penalties
* reliability evolution
* self-healing adaptation

## Impact

Enabled:

* resilient consensus
* fault-tolerant orchestration
* operational survivability

---

# 5. Byzantine Evaluation Files

## Purpose

Evaluate survivability under corrupted internal agents.

## Added Scenarios

* trust corruption
* graph poisoning
* cloud corruption
* distributed Byzantine failure

## Impact

Provided:

* resilience realism
* dependable systems evaluation
* operational robustness analysis

---

# 6. LLM Orchestration Files

## Purpose

Provide explainable cyber-defense orchestration.

## Added Components

* Ollama integration
* adaptive response reasoning
* policy generation
* trust-aware triage

## Impact

Enabled:

* explainable orchestration
* operational response intelligence
* adaptive resilience planning

---

# Final Experimental Results

ARES-V2X achieved:

* strong binary detection performance
* competitive multiclass performance
* adaptive resilience under drift
* survivability under corruption
* scalable edge deployment behavior
* explainable policy orchestration

The framework demonstrated:

* resilience-aware orchestration
* dependable cyber defense
* trust-aware distributed reasoning
* operational survivability

rather than only:

* raw classification superiority

This distinction is critical.

---

# Key Research Contributions

ARES-V2X contributes:

1. Adaptive resilient edge-cloud orchestration for V2X cyber defense.

2. Trust-aware graph propagation for distributed anomaly reasoning.

3. Byzantine-resilient adaptive consensus fusion.

4. Self-healing reliability adaptation under temporal drift.

5. LLM-assisted explainable operational response orchestration.

6. Edge deployment-aware cyber-resilience evaluation.

7. Multiclass and binary V2X attack analysis under realistic corruption scenarios.

---

# What Makes ARES-V2X Different

ARES-V2X is NOT positioned as:

* the highest-F1 classifier
* a pure ML benchmark
* a standard intrusion detector

ARES-V2X is positioned as:

# “a dependable cyber-resilience orchestration framework for distributed V2X environments.”
---

#  Research Positioning

ARES-V2X demonstrates that:

* dependable V2X security requires adaptive orchestration
* distributed trust must evolve dynamically
* resilience matters more than isolated F1-score improvements
* explainable orchestration improves operational awareness
* edge-cloud coordination is essential for realistic deployment
* survivability under corruption is critical for intelligent transportation systems

The framework therefore shifts V2X cybersecurity research from:

"static attack classification"

to:

"adaptive cyber-resilience orchestration under unreliable distributed conditions."

---
# Final Notes

ARES-V2X evolved through multiple redesign stages.

The project initially started as a multiclass V2X detection benchmark.

However, the final framework became:

* resilience-aware
* trust-aware
* deployment-aware
* graph-aware
* drift-aware
* Byzantine-aware
* edge-cloud coordinated
* explainable
* agentic

The strongest contribution is not pure classification.

The strongest contribution is:

# dependable cyber-resilience orchestration for distributed V2X security.

