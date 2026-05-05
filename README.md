# DABC-FJSP-XAI

This repository contains the implementation for the master's thesis:

**Artificial Bee Colony Algorithm for Job Scheduling with XAI Visualization**

---

## Research Objective

This study applies the Discrete Artificial Bee Colony (DABC) algorithm to the Flexible Job Shop Scheduling Problem (FJSP), and integrates Explainable Artificial Intelligence (XAI) methods to analyze:

- Optimization performance  
- Search behavior  
- Scheduling decision impact  

---

## Methodology

The research framework consists of three main components:

### 1. DABC Optimization
- Solves FJSP using OS + MA encoding  
- Minimizes makespan (Cmax)  

### 2. Search Behavior Explanation (Decision Tree)
- Explains when and why operations (swap, insert, reassign) are applied  

### 3. Solution-level Explanation (ANN + SHAP)
- Uses ANN as a surrogate model  
- Applies SHAP to interpret feature importance  

---

## Project Structure

### dabc_fjsp.py
Main DABC algorithm, including:
- 30 runs experiment  
- Convergence analysis  
- Gantt chart  
- Dataset generation  

### baseline_fjsp.py
Baseline comparison methods:
- Random Search (RS)  
- SPT + ECM  
- MWR + ECM  

### ann_shap_pos_ma.py
ANN + SHAP analysis using:
- POS (operation position)  
- MA (machine assignment)  

### ann_shap_ma_only.py
ANN + SHAP analysis using:
- MA only (machine assignment)  

---

## Benchmark

Kacem 4×5 Flexible Job Shop Scheduling Problem  

---

## Key Findings

- DABC achieves high-quality solutions (BKS = 11)  
- Machine assignment plays a dominant role in makespan  
- POS+MA model captures additional scheduling effects  
- SHAP effectively identifies critical operations  

---

## Author

Shih-Chia Yeh  
National Yang Ming Chiao Tung University  
Department of Industrial Engineering and Management
