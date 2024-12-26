# Bayesian Network for Fare Classification

## 📚 **Project Overview**
This project focuses on developing a Bayesian network model for fare classification in a public transportation system. Using a dataset containing information about bus routes, stops, distances, and fare categories, we construct and evaluate three models:
1. An **Initial Bayesian Network**
2. A **Pruned Bayesian Network**
3. An **Optimized Bayesian Network**

Each model is tested on a validation dataset, and their performance is compared based on accuracy, runtime, and efficiency.

---

## 🚀 **Objective**
- Build an initial Bayesian network for fare classification.
- Apply pruning techniques to simplify and enhance efficiency.
- Optimize the network using structure refinement methods.
- Compare the models based on accuracy, runtime, and efficiency.
- Return `.pkl` files for all three models.

---

## 🧠 **Dataset Features**
The Bayesian network uses the following features:
- **Start Stop ID (S):** Stop ID where the journey begins.
- **End Stop ID (E):** Stop ID where the journey ends.
- **Distance (D):** Distance between start and end stops.
- **Zones Crossed (Z):** Number of fare zones crossed.
- **Route Type (R):** Type of route (e.g., standard, express).
- **Fare Category (F):** Target variable classified as Low, Medium, or High.

---

## 🛠️ **Tasks**
### **Task 1: Initial Bayesian Network Construction**
- Construct the Bayesian network using the specified features.
- Ensure dependencies between relevant feature pairs.
- Visualize the initial Bayesian network structure.
- Evaluate and record runtime and accuracy.

### **Task 2: Pruned Bayesian Network**
- Apply pruning techniques:
   - Edge Pruning
   - Node Pruning
   - Conditional Probability Table (CPT) simplification
- **Method Used:** Independence Testing via `bn.independence_test` with `prune=True`. Edges failing the statistical significance test (alpha=0.05) are removed.
   - Independence tests use statistical methods like Chi-Square for categorical variables.
   - Strong evidence (p-value ≤ 0.05) indicates a meaningful connection; otherwise, edges are removed.
- **Results:**
   - **Edge Reduction:** 15 → 10 edges (33.33% reduction)
   - **Fit Time Comparison:**
      - Improvement: Approx. 12.2% reduction in fitting time
- **Improvements:**
   1. **Efficiency:** Reduced computational time and improved inference speed.
   2. **Simplification:** Statistically significant edges are retained, reducing overfitting risks.
   3. **Potential Accuracy Improvement:** The model is less likely to overfit, improving generalization.
- Visualize the pruned Bayesian network.

### **Task 3: Optimized Bayesian Network**
- Apply optimization techniques such as **Structure Learning (e.g., Hill Climbing)**.
- **Method Used:** Hill Climbing Algorithm with BIC (Bayesian Information Criterion) as the scoring metric.
   - Iteratively adds, removes, or reverses edges to minimize the BIC score.
- **Results:**
   - **Edge Reduction:** 15 → 4 edges (73.33% reduction)
   - **Fit Time Comparison:**
      - Improvement: Approx. 98.82% reduction in fitting time
- **Improvements:**
   1. **Efficiency:** Significant reduction in fitting and inference time.
   2. **Simplification:** A clearer structure focusing on the most important relationships.
   3. **Generalization:** Reduced overfitting risk and potential accuracy improvement.
- Visualize the optimized Bayesian network.

---

## 📊 **Evaluation Metrics**
- **Accuracy:** Measure prediction correctness on the validation dataset.
- **Runtime:** Record time taken for initialization and training.
- **Model Complexity:** Analyze network structure and efficiency.

---

## 📈 **Results Visualization**
- **Graph Visualizations:** Three Graphviz PNGs showing:
   1. Initial Bayesian Network
   2. Pruned Bayesian Network
   3. Optimized Bayesian Network
- Comparative analysis of accuracy, runtime, and efficiency.
- Observations and conclusions documented.

---


## ⚙️ **Setup and Usage**

## **Virtual Environment Setup (Recommended)**

To ensure a clean and isolated development environment, it is highly recommended to use a **virtual environment** (`venv`) for this project.  

### 🛠️ **Steps to Set Up `venv`:**

1. **Create a Virtual Environment:**  
   ```bash
   python -m venv venv
2. **Activate the Virtual Environment:**
   - **On Windows:**
    ```cmd
    .\venv\Scripts\activate
    ```
   - **On macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Deactivate the Environment (When Done):**
    ```bash
    deactivate
    ```
5. Run the project:
   ```bash
   python FareClassification.py
   ```
6. Ensure `.pkl` files for each model are returned.

*Note: By using venv, you can avoid dependency conflicts and maintain consistency across different development setups.*

---

## 📝 **Conclusion**
- Compared the efficiency and accuracy of the three Bayesian networks.
- Highlighted the impact of pruning and optimization on performance.
- Provided key insights into Bayesian network design for fare classification.

---

## 📬 **Contact**
For questions or collaboration, feel free to reach out!

---

Happy Coding! 🚀


