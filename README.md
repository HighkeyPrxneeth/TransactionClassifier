# TransactionClassifier
### Automated AI-Based Financial Transaction Categorisation

**Theme:** End-to-End Autonomous Categorisation  
**Status:** Prototype / Hackathon Submission

---

## Overview

**TransactionClassifier** is a high-performance, privacy-first AI system designed to classify raw financial transaction strings (e.g., "STARBUCKS 0042", "AMZN MKTP US") into hierarchical user-defined categories. 

Unlike traditional solutions that rely on expensive, high-latency third-party APIs, our system runs **entirely locally**. It leverages a novel "Trajectory Learning" approach—combining pre-trained semantic embeddings with a custom deep learning head—to achieve business-grade accuracy without data ever leaving your infrastructure.

## Key Features

*   **100% Local & Autonomous:** No external API calls. All inference happens within your environment, ensuring zero data leakage and minimal latency.
*   **Taxonomy Agnostic:** The category tree is **not hardcoded**. You can update the taxonomy dynamically via the API configuration without retraining the model.
*   **Modern Architecture:** Built on a custom `ModernTrajectoryNet` utilizing state-of-the-art components like **SwiGLU**, **RMSNorm**, and **Squeeze-and-Excitation** blocks for maximum expressivity.
*   **Robust & Resilient:** Trained with noise injection and masking augmentations to handle messy, real-world transaction data.
*   **Hybrid Inference Engine:** Combines deep semantic understanding with a keyword-boosting heuristic layer to ensure high confidence even for edge cases.

---

## Technical Approach

Our solution moves beyond simple keyword matching or standard text classification. We treat transaction classification as a **trajectory mapping problem** in a high-dimensional semantic space.

### 1. The "Trajectory" Concept
Raw transactions and clean categories often live in different semantic "subspaces" (e.g., "UBER *TRIP" vs "Transportation").
1.  **Embedding:** We use a `SentenceTransformer` (e.g., `all-MiniLM-L6-v2`) to project both the transaction and the target taxonomy into a shared vector space.
2.  **Trajectory Mapping:** Our custom **ModernTrajectoryNet** learns a non-linear transformation to "push" the noisy transaction embedding closer to its clean, hierarchical category embedding.
3.  **Similarity Search:** We perform a cosine similarity search between the transformed transaction and the dynamic taxonomy embeddings to find the best match.

### 2. Model Architecture
The core model (`model.py`) is designed for efficiency and stability:
*   **Pre-Norm Architecture** with **RMSNorm** for training stability.
*   **SwiGLU Activations** for superior feed-forward performance.
*   **SEBlocks (Squeeze-and-Excitation)** for dynamic channel-wise attention.
*   **Stochastic Depth (DropPath)** for regularization during training.

### 3. Data Strategy
*   **Synthetic & Public Data:** We utilize a combination of public financial datasets and synthetically generated variations to ensure broad coverage.
*   **Augmentation:** During training, we apply random noise and token masking to embeddings, forcing the model to learn robust features rather than memorizing specific strings.

---

## Evaluation & Metrics

We focus on rigorous evaluation metrics to ensure reliability:

*   **Macro F1-Score:** Our primary target metric to ensure balanced performance across all categories, not just the frequent ones.
*   **Monotonicity:** A custom metric that measures if the model's "trajectory" consistently moves closer to the target category layer-by-layer.
*   **LastSim:** The cosine similarity between the final output and the ground truth category embedding.

*(Note: Detailed confusion matrices and per-class F1 scores are generated during the evaluation phase.)*

---

## Installation & Usage

### Prerequisites
*   Python 3.10+
*   CUDA-capable GPU (recommended for training)

### Setup
```bash
# Clone the repository
git clone https://github.com/HighkeyPrxneeth/TransactionClassifier.git
cd TransactionClassifier

# Install dependencies
pip install -r requirements.txt
```

### Training
To train the model on your dataset:
```bash
python train.py
```
*Configuration parameters (batch size, learning rate, etc.) can be modified in `config.py`.*

### Running the API (Inference)
Start the FastAPI server:
```bash
uvicorn app:app --reload
```

### Example Request
**POST** `http://127.0.0.1:8000/predict`

```json
{
  "transaction": "UBER EATS SAN FRANCISCO CA",
  "taxonomy_text": "Food\n    Dining\n    Groceries\nTransport\n    Rideshare\n    Fuel"
}
```

**Response:**
```json
{
  "matches": [
    {
      "category": "Food > Dining",
      "score": 0.92
    },
    {
      "category": "Transport > Rideshare",
      "score": 0.15
    }
  ]
}
```

---

## Roadmap (Finale Phase)

While the current system meets the core objectives of the problem statement, we have planned the following enhancements for the final production release:

*   **Explainability UI:** A visual dashboard showing which tokens in the transaction contributed most to the decision.
*   **Active Learning Loop:** A feedback endpoint where users can correct low-confidence predictions, which are then queued for the next training batch.
*   **Bias Mitigation:** Comprehensive analysis of model performance across different merchant regions to ensure fairness.
*   **Throughput Benchmarks:** Formal stress testing reports for high-volume batch processing.

---


