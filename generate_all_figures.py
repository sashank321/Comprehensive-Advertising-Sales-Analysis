import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from pandas.plotting import scatter_matrix

def main():
    # Ensure images directory exists in CWD (which will be the project root)
    os.makedirs('images', exist_ok=True)
    
    # Load dataset
    df = pd.read_csv('Dataset/advertising.csv')
    print('Dataset Loaded Successfully. Dimensions:', df.shape)
    
    # --- Figure 1 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Figure 1: IQR Outlier Boundary Check for Features', fontsize=14, fontweight='bold')
    cols = ['TV', 'Radio', 'Newspaper']
    for i, col in enumerate(cols):
        ax = axes[i]
        ax.boxplot(df[col], orientation='horizontal', patch_artist=True, boxprops=dict(facecolor='#abc9e9'))
        ax.set_title(f'{col} Spend Distribution')
        ax.set_xlabel('Promotional Expenditure ($k)')
        ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig1_outlier_boxplots.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Figure 2: Feature and Target Distribution Analysis', fontsize=16, fontweight='bold')
    cols = ['TV', 'Radio', 'Newspaper', 'Sales']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, col in enumerate(cols):
        ax = axes[i//2, i%2]
        ax.hist(df[col], bins=15, color=colors[i], edgecolor='black', alpha=0.7, density=True)
        mu, std = stats.norm.fit(df[col])
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'k--', linewidth=2, label=rf'Fit: \n$\mu={mu:.1f}$, $\sigma={std:.1f}$')
        ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Spend / Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('images/fig2_distributions.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 3 ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Figure 3: Feature and Target Spreads Boxplots', fontsize=14, fontweight='bold')
    cols = ['TV', 'Radio', 'Newspaper', 'Sales']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, col in enumerate(cols):
        ax = axes[i]
        ax.boxplot(df[col], patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.6))
        ax.set_title(f'{col} Spread')
        ax.set_ylabel('Value Scale')
        ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig3_boxplots_vertical.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 4 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Figure 4: Bivariate Scatter Analysis (Features vs. Sales)', fontsize=16, fontweight='bold')
    features = ['TV', 'Radio', 'Newspaper']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, feat in enumerate(features):
        ax = axes[i]
        ax.scatter(df[feat], df['Sales'], color=colors[i], edgecolor='k', alpha=0.7)
        m, c = np.polyfit(df[feat], df['Sales'], 1)
        ax.plot(df[feat], m*df[feat] + c, color='red', linestyle='-', linewidth=2, label=f'y = {m:.3f}x + {c:.2f}')
        ax.set_title(f'{feat} vs Sales', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{feat} Advertising Spend ($k)')
        ax.set_ylabel('Sales (Units in Thousands)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('images/fig4_scatter_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 5 ---
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Figure 5: Correlation Analytics with Sales', fontsize=14, fontweight='bold')
    ax_heat = axes[0]
    cax = ax_heat.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax_heat)
    ax_heat.set_xticks(np.arange(len(corr_matrix.columns)))
    ax_heat.set_yticks(np.arange(len(corr_matrix.columns)))
    ax_heat.set_xticklabels(corr_matrix.columns)
    ax_heat.set_yticklabels(corr_matrix.columns)
    plt.setp(ax_heat.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            ax_heat.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}', ha='center', va='center', color='black', fontweight='bold')
    ax_heat.set_title('Correlation Analysis Heatmap')
    ax_bar = axes[1]
    sales_corr = corr_matrix['Sales'].drop('Sales')
    ax_bar.bar(sales_corr.index, sales_corr.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', alpha=0.8)
    ax_bar.set_ylim(-1, 1)
    ax_bar.axhline(0, color='black', linewidth=1)
    ax_bar.set_title('Correlation Strength with Sales')
    ax_bar.set_ylabel('Pearson Correlation Coefficient')
    for idx, val in enumerate(sales_corr.values):
        ax_bar.text(idx, val + (0.05 if val > 0 else -0.1), f'{val:.4f}', ha='center', fontweight='bold')
    ax_bar.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig5_correlation.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 6 ---
    fig, ax = plt.subplots(figsize=(12, 12))
    scatter_matrix(df.select_dtypes(include=[np.number]), alpha=0.8, figsize=(10, 10), ax=ax, diagonal='kde', color='#1f77b4')
    plt.suptitle('Figure 6: Pairwise Scatter Matrix and Kernel Density Estimations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/fig6_scatter_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 7 ---
    sales_mean = df['Sales'].mean()
    sales_std = df['Sales'].std()
    threshold = 15.0
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(sales_mean - 4*sales_std, sales_mean + 4*sales_std, 500)
    p = stats.norm.pdf(x, sales_mean, sales_std)
    ax.plot(x, p, 'k-', linewidth=3, label='Theoretical PDF')
    ax.fill_between(x, p, where=(x > threshold), color='red', alpha=0.3, label=f'P(Sales > {threshold}k)')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold}k')
    ax.set_title('Figure 7: Normal Distribution Fit and Critical Threshold Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sales Revenue (Units in Thousands)')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig7_probability_threshold.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 8 ---
    df['TV_Bin'] = pd.qcut(df['TV'], q=3, labels=['Low', 'Medium', 'High'])
    g_low = df[df['TV_Bin'] == 'Low']['Sales']
    g_med = df[df['TV_Bin'] == 'Medium']['Sales']
    g_high = df[df['TV_Bin'] == 'High']['Sales']
    categories = ['Low TV Spend', 'Medium TV Spend', 'High TV Spend']
    means = [g_low.mean(), g_med.mean(), g_high.mean()]
    errors = [
        stats.sem(g_low) * stats.t.ppf(0.975, len(g_low)-1),
        stats.sem(g_med) * stats.t.ppf(0.975, len(g_med)-1),
        stats.sem(g_high) * stats.t.ppf(0.975, len(g_high)-1)
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(categories, means, yerr=errors, fmt='o', color='crimson', ecolor='darkblue', elinewidth=3, capsize=10, markersize=10, label='Mean Sales & 95% CI')
    ax.set_title('Figure 8: Mean Sales with 95% Confidence Intervals across TV Spend Groups', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sales (Units in Thousands)')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig('images/fig8_confidence_intervals.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- MLR Model Training ---
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlr_start_train = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    mlr_end_train = time.time()
    mlr_train_time = mlr_end_train - mlr_start_train
    coefficients = model.coef_
    coef_names = X.columns
    
    mlr_start_pred = time.time()
    y_pred = model.predict(X_test)
    mlr_end_pred = time.time()
    mlr_pred_time = mlr_end_pred - mlr_start_pred
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)
    residuals = y_test - y_pred

    # --- Figure 9 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Figure 9: Model Diagnostics & Residual Analysis', fontsize=16, fontweight='bold')
    axes[0, 0].scatter(y_pred, residuals, color='purple', edgecolor='k', alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Residuals vs. Predicted Values')
    axes[0, 0].set_xlabel('Predicted Sales (k units)')
    axes[0, 0].set_ylabel('Residual (Error)')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    
    axes[0, 1].hist(residuals, bins=10, color='teal', edgecolor='black', alpha=0.7, density=True)
    res_mu, res_std = stats.norm.fit(residuals)
    xmin, xmax = axes[0, 1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, res_mu, res_std)
    axes[0, 1].plot(x, p, 'k--', linewidth=2, label=rf'Fit: \n$\mu={res_mu:.2f}$, $\sigma={res_std:.2f}$')
    axes[0, 1].set_title('Histogram of Residuals')
    axes[0, 1].set_xlabel('Residual value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    
    stats.probplot(residuals, dist='norm', plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set_markerfacecolor('blue')
    axes[1, 0].get_lines()[0].set_alpha(0.7)
    axes[1, 0].get_lines()[1].set_color('red')
    axes[1, 0].set_title('Normal Q-Q Plot of Residuals')
    axes[1, 0].grid(True, linestyle=':', alpha=0.5)
    
    error_pct = (residuals / y_test) * 100
    axes[1, 1].hist(error_pct, bins=10, color='orange', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Percentage Prediction Error Distribution')
    axes[1, 1].set_xlabel('Percentage Error (%)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig9_mlr_diagnostics.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 10 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='#1f77b4', edgecolor='k', alpha=0.8, s=80, label='Observations')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
    ax.set_title('Figure 10: Actual vs. Predicted Sales Volume', fontsize=14, fontweight='bold')
    ax.set_xlabel('Actual Sales (Units in Thousands)')
    ax.set_ylabel('Predicted Sales (Units in Thousands)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig10_actual_vs_predicted.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 11 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coef_names, coefficients, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', alpha=0.8, height=0.5)
    ax.set_title('Figure 11: Feature Importance (Regression Coefficients)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Coefficient (Beta Value)')
    ax.set_ylabel('Advertising Medium')
    for idx, val in enumerate(coefficients):
        ax.text(val + 0.002, idx, f'{val:.4f}', va='center', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig11_feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- ANN Model Training ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ann_model = MLPRegressor(
        hidden_layer_sizes=(8, 4),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42,
        early_stopping=False
    )
    ann_start_train = time.time()
    ann_model.fit(X_train_scaled, y_train)
    ann_end_train = time.time()
    ann_train_time = ann_end_train - ann_start_train
    
    # --- Figure 12 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ann_model.loss_curve_, color='navy', linewidth=2, label='Training Loss (MSE)')
    ax.set_title('Figure 12: Neural Network Loss Convergence Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch Iteration')
    ax.set_ylabel('Loss (Mean Squared Error)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig12_ann_loss_curve.png', bbox_inches='tight', dpi=300)
    plt.close()

    ann_start_pred = time.time()
    y_pred_ann = ann_model.predict(X_test_scaled)
    ann_end_pred = time.time()
    ann_pred_time = ann_end_pred - ann_start_pred

    ann_mae = metrics.mean_absolute_error(y_test, y_pred_ann)
    ann_mse = metrics.mean_squared_error(y_test, y_pred_ann)
    ann_rmse = np.sqrt(ann_mse)
    ann_r2 = metrics.r2_score(y_test, y_pred_ann)
    ann_residuals = y_test - y_pred_ann

    # --- Figure 13 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Figure 13: Neural Network Model Diagnostics & Residual Analysis', fontsize=16, fontweight='bold')
    axes[0, 0].scatter(y_pred_ann, ann_residuals, color='magenta', edgecolor='k', alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Residuals vs. Predicted Values')
    axes[0, 0].set_xlabel('Predicted Sales (k units)')
    axes[0, 0].set_ylabel('Residual (Error)')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    
    axes[0, 1].hist(ann_residuals, bins=10, color='darkgreen', edgecolor='black', alpha=0.7, density=True)
    ann_res_mu, ann_res_std = stats.norm.fit(ann_residuals)
    xmin, xmax = axes[0, 1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, ann_res_mu, ann_res_std)
    axes[0, 1].plot(x, p, 'k--', linewidth=2, label=rf'Fit: \n$\mu={ann_res_mu:.2f}$, $\sigma={ann_res_std:.2f}$')
    axes[0, 1].set_title('Histogram of Residuals')
    axes[0, 1].set_xlabel('Residual value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    
    stats.probplot(ann_residuals, dist='norm', plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set_markerfacecolor('magenta')
    axes[1, 0].get_lines()[0].set_alpha(0.7)
    axes[1, 0].get_lines()[1].set_color('red')
    axes[1, 0].set_title('Normal Q-Q Plot of Residuals')
    axes[1, 0].grid(True, linestyle=':', alpha=0.5)
    
    axes[1, 1].scatter(y_test, y_pred_ann, color='darkorange', edgecolor='k', alpha=0.8, s=80, label='Observations')
    lims = [min(y_test.min(), y_pred_ann.min()), max(y_test.max(), y_pred_ann.max())]
    axes[1, 1].plot(lims, lims, color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
    axes[1, 1].set_title('Actual vs. Predicted Sales Volume')
    axes[1, 1].set_xlabel('Actual Sales (Units in Thousands)')
    axes[1, 1].set_ylabel('Predicted Sales (Units in Thousands)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig13_ann_diagnostics.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 14: Model Performance Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Figure 14: Model Performance Comparison — MLR vs. ANN', fontsize=16, fontweight='bold')
    labels = ['MAE', 'RMSE']
    mlr_errors = [mae, rmse]
    ann_errors = [ann_mae, ann_rmse]
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width/2, mlr_errors, width, label='Multiple Linear Regression', color='#1f77b4', edgecolor='black', alpha=0.8)
    axes[0].bar(x + width/2, ann_errors, width, label='Artificial Neural Network', color='#2ca02c', edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('Error (k units)')
    axes[0].set_title('Prediction Error Comparison (MAE & RMSE)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.5)
    for i, val in enumerate(mlr_errors):
        axes[0].text(i - width/2, val + 0.03, f'{val:.4f}', ha='center', fontweight='bold')
    for i, val in enumerate(ann_errors):
        axes[0].text(i + width/2, val + 0.03, f'{val:.4f}', ha='center', fontweight='bold')

    r2_labels = ['MLR', 'ANN']
    r2_values = [r2, ann_r2]
    bars = axes[1].bar(r2_labels, r2_values, color=['#1f77b4', '#2ca02c'], edgecolor='black', alpha=0.8, width=0.5)
    axes[1].set_ylabel('R² Score')
    axes[1].set_ylim(0.85, 0.95)
    axes[1].set_title('R² Score Comparison (Variance Explained)')
    axes[1].grid(True, linestyle=':', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/fig14_model_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 15: Actual vs. Predicted Sales Side-by-Side ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True, sharex=True)
    fig.suptitle('Figure 15: Actual vs. Predicted Sales — Model Comparison', fontsize=16, fontweight='bold')
    lims = [min(y_test.min(), y_pred.min(), y_pred_ann.min()) - 1, max(y_test.max(), y_pred.max(), y_pred_ann.max()) + 1]
    
    axes[0].scatter(y_test, y_pred, color='#1f77b4', edgecolor='k', alpha=0.8, s=80, label='MLR Predictions')
    axes[0].plot(lims, lims, color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
    axes[0].set_title('Multiple Linear Regression')
    axes[0].set_xlabel('Actual Sales (Units in Thousands)')
    axes[0].set_ylabel('Predicted Sales (Units in Thousands)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.5)
    
    axes[1].scatter(y_test, y_pred_ann, color='#2ca02c', edgecolor='k', alpha=0.8, s=80, label='ANN Predictions')
    axes[1].plot(lims, lims, color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
    axes[1].set_title('Artificial Neural Network')
    axes[1].set_xlabel('Actual Sales (Units in Thousands)')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig15_side_by_side_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    # --- Figure 16: Residual Distribution Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Figure 16: Residual Distribution Comparison — MLR vs. ANN', fontsize=16, fontweight='bold')
    axes[0].hist(residuals, bins=12, color='#1f77b4', edgecolor='black', alpha=0.5, label='MLR Residuals', density=True)
    axes[0].hist(ann_residuals, bins=12, color='#2ca02c', edgecolor='black', alpha=0.5, label='ANN Residuals', density=True)
    
    xmin, xmax = axes[0].get_xlim()
    x_axis = np.linspace(xmin, xmax, 100)
    p_mlr = stats.norm.pdf(x_axis, *stats.norm.fit(residuals))
    p_ann = stats.norm.pdf(x_axis, *stats.norm.fit(ann_residuals))
    axes[0].plot(x_axis, p_mlr, color='darkblue', linestyle='--', linewidth=2, label='MLR Fit')
    axes[0].plot(x_axis, p_ann, color='darkgreen', linestyle='-', linewidth=2, label='ANN Fit')
    axes[0].set_title('Residual Density & Normal Fits')
    axes[0].set_xlabel('Residual Error (k units)')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.5)
    
    axes[1].boxplot([residuals, ann_residuals], patch_artist=True, tick_labels=['MLR Residuals', 'ANN Residuals'],
                    boxprops=dict(facecolor='#abc9e9', alpha=0.7), medianprops=dict(color='red', linewidth=2))
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('Residual Spread Comparison')
    axes[1].set_ylabel('Error Scale')
    axes[1].grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/fig16_residual_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("All 16 figures generated and saved successfully in the 'images' directory.")

if __name__ == '__main__':
    main()
