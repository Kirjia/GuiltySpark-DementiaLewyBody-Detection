# GulitySpark-DementiaLewyBody-Detection
# 🧠 AI for Neurodegenerative Diseases Detection
**Differential Diagnosis of Alzheimer's Disease and Lewy Body Dementia using Clinical Tabular Data**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-blue)](https://xgboost.ai/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This project aims to tackle a highly complex clinical challenge: the differential diagnosis between **Alzheimer's Disease (AD)**, **Lewy Body Dementia (DLB)**, and **Cognitively Normal (Healthy)** patients. 

Leveraging the extensive longitudinal clinical dataset from the **National Alzheimer's Coordinating Center (NACC)**, this repository implements both state-of-the-art tree-based models and a custom Deep Learning architecture specifically engineered for tabular clinical data.

### 🎯 Key Objectives
* Overcome the curse of dimensionality and feature cross-talk in tabular datasets.
* Handle severe class imbalances inherent to rare neurodegenerative diseases (DLB).
* Provide rigorous model calibration metrics (ECE L2) to assess clinical reliability.
* Extract deep feature importance using Neural Network Self-Attention mechanisms.

---

## 🔬 Methodology & Architectures

The project explores multiple machine learning paradigms, culminating in a custom deep learning architecture.

### 1. Traditional ML & Ensembles
* **Random Forest**: Baseline for tree-based bagging.
* **XGBoost**: Highly calibrated boosting model, achieving the best overall balance between calibration (ECE L2: 0.0293) and discriminative power.
* **Support Vector Machine (SVM)**: Evaluated for hyperplane separation, revealing the complex, non-linear overlap between AD and DLB.

### 2. Sparse Autoencoder (Deep Learning)
A custom Neural Network engineered in TensorFlow/Keras to extract stable biomarkers while resisting memorization.
* **Input Dropout (On-The-Fly Augmentation)**: Random masking of 10% of clinical features per epoch to force anti-fragile learning.
* **Dynamic Batch Resampling (`tf.data`)**: A continuous stochastic engine to perfectly balance classes dynamically at the batch level during training.
* **Sparse Bottleneck**: Dimensionality reduction (16 neurons) using ReLU and $L1$ activity regularization to force the extraction of distinct clinical profiles.

---

## 📊 Dataset Notice
Due to strict privacy and data usage agreements, **the NACC dataset cannot be shared in this repository**. 
To run this code, you must formally request access to the UDS (Uniform Data Set) via the [NACC official website](https://naccdata.org/). 

*The preprocessing scripts provided in `src/` assume the presence of the raw NACC CSV file in the `data/raw/` directory.*

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.11+
* CUDA-enabled GPU (Highly recommended for Autoencoder training)

### Setup Environment
1. Clone the repository:
   ```bash
   git clone [https://github.com/GuiltySpark343/DementiaLewyBody-Detection.git](https://github.com/GuiltySpark343/DementiaLewyBody-Detection.git)
   cd DementiaLewyBody-Detection


Create and activate a virtual environment:

```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
```
  Install dependencies:

```bash
  pip install -r requirements.txt
```
# Execution
The complete pipeline (Preprocessing -> EDA -> Classical ML -> Deep Learning) can be executed sequentially via the provided Jupyter Notebooks in the notebooks/ directory.

📈 Key Results
XGBoost emerged as the most calibrated model, perfectly isolating Healthy patients and achieving an F1-Score of 85.92% with a remarkable ECE L2 of 0.0293.

The Sparse Autoencoder demonstrated the highest sensitivity toward the minority class (Lewy Body Dementia), maximizing True Positives and minimizing missed diagnoses and Dynamic Resampling methodologies.


✍️ Author
Vincenzo Danese (GitHub: @GuiltySpark343)

Computer Science Department, University of Salerno, Fisciano, Italy.

📜 Acknowledgments
Special thanks to the NACC (National Alzheimer's Coordinating Center) and the NIA/NIH for funding and maintaining the database (Grant U24 AG072122) that made this research possible.



<ElicitationsGroup message="Come vuoi proseguire?">
<Elicitation label="Genera il file requirements.txt" query="Genera il file requirements.txt" query_intent="CLICKABLE_SUGGESTION" />
<Elicitation label="Scrivi il file .gitignore per il progetto" query="Scrivi il file .gitignore per il progetto" query_intent="CLICKABLE_SUGGESTION" />
<Elicitation label="Crea uno script Python per automatizzare il training" query="Crea uno script Python per automatizzare il training" query_intent="CLICKABLE_SUGGESTION" />
</ElicitationsGroup>
