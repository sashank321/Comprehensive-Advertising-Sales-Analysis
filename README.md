# Comprehensive Advertising Sales Analysis Using Multiple Linear Regression and Artificial Neural Networks

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.0-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-2.2.3-navy?style=for-the-badge&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.0-teal?style=for-the-badge&logo=python)
![Playwright](https://img.shields.io/badge/Playwright-1.49-success?style=for-the-badge&logo=playwright)

> **A Final Year Dissertation Project for the Course Mathematics for Data Science & Analytics (MDSA)**  
> **Institution:** KL University — Department of Mathematics  
> **Submitted By:** J Sasank (2510030267) | R Sai Karthik (2510030295) | B Shriyan (2510030057)  
> **Supervised By:** Dr. M. Nagesh  

---

## 📌 Project Overview

In the modern enterprise landscape, marketing expenditure is no longer viewed as a speculative operational expense but as a high-stakes capital investment requiring rigorous empirical justification. This project presents an advanced, dual-model econometric and machine learning investigation into the causal and predictive relationships between multi-channel promotional expenditures (**TV**, **Radio**, and **Newspaper**) and resulting product **Sales volume**.

To achieve a holistic balance between **parametric interpretability** and **non-linear predictive power**, this study establishes two distinct modeling paradigms:
1. **Multiple Linear Regression (MLR):** A classical econometric model establishing baseline linear coefficients, statistical significance ($p$-values), and structural confidence intervals.
2. **Artificial Neural Networks (ANN):** A robust Multi-Layer Perceptron (MLPRegressor) capable of capturing complex, higher-order non-linear feature interactions and cross-channel marketing synergies.

---

## 📊 Dataset Description

The investigation utilizes the benchmark `Advertising.csv` dataset, comprising 200 empirical observations across diverse regional markets. 

| Feature | Data Type | Measurement Scale | Description |
| :--- | :--- | :--- | :--- |
| **TV** | `float64` | Promotional Budget ($k) | Advertising capital allocated to broadcast television networks. |
| **Radio** | `float64` | Promotional Budget ($k) | Advertising capital allocated to broadcast radio stations. |
| **Newspaper** | `float64` | Promotional Budget ($k) | Advertising capital allocated to print newspaper publications. |
| **Sales** | `float64` | Volume (Units in Thousands) | **[Target Variable]** Resulting product sales observed in the market. |

*Note: The dataset underwent comprehensive exploratory data analysis (EDA), including Z-score/IQR outlier checks, kernel density estimations, and Pearson correlation matrices prior to model construction.*

---

## 🔬 Model Performance & Comparison

Both models were trained and evaluated on an identical `80:20` train-test split ($n_{train}=160$, $n_{test}=40$) using a fixed random state (`random_state=42`) to guarantee absolute empirical reproducibility.

| Performance Metric | Multiple Linear Regression (MLR) | Artificial Neural Network (ANN) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **R² Score (Variance Explained)** | `0.8994` | `0.9095` | **+1.01%** |
| **Root Mean Squared Error (RMSE)** | `1.7816` | `1.6896` | **-5.16%** |
| **Mean Absolute Error (MAE)** | `1.4608` | `1.3412` | **-8.19%** |
| **Execution Training Time** | `0.0042 s` | `0.4518 s` | Base vs. Iterative |

### 🔍 Key Takeaways from Model Comparison:
- **Explanatory Breadth:** Both models demonstrate exceptional predictive accuracy, capturing approximately ~90% to 91% of total sales variance.
- **Superiority of ANN:** The Artificial Neural Network achieves notable reductions in both MAE (**-8.19%**) and RMSE (**-5.16%**), confirming the presence of non-linear synergies between broadcast advertising channels that classical linear planes cannot fully capture.
- **Residual Boundedness:** Residual diagnostics confirm that both models adhere to strict normal error distributions centred near zero, confirming the absence of severe heteroscedasticity or structural bias.

---

## 📈 Key Business & Marketing Insights

### 1. The Dominance of Broadcast Television (TV)
- **Insight:** Television expenditure exhibits a commanding linear correlation with sales volume ($r = 0.7822$).
- **Impact:** For every additional `$1,000` invested in TV advertising, baseline sales increase by **45.8 units** ($\beta_{TV} = 0.0458$). TV represents the primary growth engine for product scaling.

### 2. The Multiplier Effect of Radio Advertising
- **Insight:** While Radio maintains a lower aggregate budget allocation, it demonstrates the highest relative efficiency per dollar spent ($\beta_{Radio} = 0.1890$).
- **Impact:** Each `$1,000` allocated to Radio expands sales by **189.0 units**. Furthermore, the neural network identifies strong cross-channel amplification when Radio is deployed concurrently with TV campaigns.

### 3. The Inefficiency of Print Newspapers
- **Insight:** Newspaper advertising demonstrates near-zero standalone correlation with sales ($r = 0.2283$) and yields a negligible, statistically insignificant regression coefficient ($\beta_{Newspaper} = 0.0028$).
- **Impact:** Investing `$1,000` in Newspaper advertising generates a marginal return of just **2.8 units**, failing to justify the capital allocation.

### 🎯 Strategic Recommendation: Budget Reallocation
To maximize Return on Investment (ROI) without increasing aggregate marketing expenditure, the enterprise should immediately execute a strategic pivot:
> **Reallocate 100% of the print Newspaper marketing budget directly into broadcast Television and Radio advertising.**

---

## 📂 Repository Structure

```text
├── Comprehensive_Advertising_Sales_Analysis.ipynb  # Main Jupyter Notebook (Complete Executable Analysis)
├── Comprehensive_Advertising_Sales_Analysis_Report.pdf # Publication-Quality 66-Page Premium Dissertation PDF
├── Dataset/
│   └── advertising.csv                             # Empirical Multi-Channel Marketing Dataset
├── images/                                         # 16 High-Resolution Publication-Quality Figures
│   ├── fig1_outlier_boxplots.png
│   ├── fig2_distributions.png
│   ├── fig3_boxplots_vertical.png
│   ├── fig4_scatter_analysis.png
│   ├── fig5_correlation.png
│   ├── fig6_scatter_matrix.png
│   ├── fig7_probability_threshold.png
│   ├── fig8_confidence_intervals.png
│   ├── fig9_mlr_diagnostics.png
│   ├── fig10_actual_vs_predicted.png
│   ├── fig11_feature_importance.png
│   ├── fig12_ann_loss_curve.png
│   ├── fig13_ann_diagnostics.png
│   ├── fig14_model_comparison.png
│   ├── fig15_side_by_side_comparison.png
│   └── fig16_residual_comparison.png
├── generate_all_figures.py                         # Standalone script to regenerate all 16 analytical figures
├── build_pdf_playwright.py                         # 3-Pass Playwright/pypdf script for premium PDF generation
├── dissertation.html                               # Core HTML Dissertation Source Document
└── README.md                                       # Executive Summary & Repository Documentation
```

---

## 🚀 Getting Started & Execution

### 1. Prerequisites & Environment Setup
Ensure you have Python 3.13+ installed along with the required scientific libraries:
```bash
pip install pandas numpy matplotlib scipy scikit-learn beautifulsoup4 playwright pypdf
```
*If utilizing the PDF rendering engine, initialize Playwright's browser binaries:*
```bash
playwright install chromium
```

### 2. Running the Core Data Science Analysis
You can open and execute the primary Jupyter Notebook directly in JupyterLab, VS Code, or Google Colab:
```bash
jupyter notebook Comprehensive_Advertising_Sales_Analysis.ipynb
```

### 3. Regenerating Publication Figures & Premium PDF Report
To autonomously re-execute the econometric pipeline, regenerate all 16 high-resolution figures into the `images/` directory, and compile the premium 66-page dissertation PDF report, run the following standalone compilation scripts:
```bash
# 1. Regenerate all analytical figures
python generate_all_figures.py

# 2. Recompile the 3-pass publication-quality PDF report
python build_pdf_playwright.py
```

---

## 📜 Academic Integrity & License
This repository contains the official, verified empirical findings and complete implementation code for the final year dissertation in **Mathematics for Data Science & Analytics (MDSA)** at **KL University**. All models, coefficients, and visualizations are directly reproducible from the provided source codebase.
