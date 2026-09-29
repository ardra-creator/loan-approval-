"""
Exploratory Data Analysis (EDA) Script for Loan Approval Prediction
Generates only the specified useful plots and saves them in static/plots/
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(csv_path="loan_approval_dataset.csv", output_dir="static/plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    print("Running EDA...")
    print(f"Dataset shape: {df.shape}")

    # Set visualization theme
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # 1. Loan Approval Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts_df = pd.DataFrame({'Loan_Status': counts.index, 'Count': counts.values})
    sns.barplot(data=counts_df, x='Loan_Status', y='Count', hue='Loan_Status', ax=ax, palette={'Y': '#2ecc71', 'N': '#e74c3c'}, legend=False)
    ax.set_title("Loan Approval Distribution (Target Variable)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Loan Status (Y: Approved, N: Not Approved)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)} ({height/len(df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 4),
                    textcoords='offset points')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "loan_approval_distribution.png"), dpi=200)
    plt.close(fig)
    print("Saved: loan_approval_distribution.png")

    # 2. Numerical Feature Distributions
    num_cols = ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Dependents']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#3498db', bins=25)
        axes[i].set_title(f"Distribution of {col}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    axes[5].axis('off')  # unused subplot
    plt.suptitle("Distributions of Key Numerical Features", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "numerical_feature_distributions.png"), dpi=200)
    plt.close(fig)
    print("Saved: numerical_feature_distributions.png")

    # 3. Correlation Heatmap for Numerical Features
    corr_cols = ['Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 'Credit_History']
    fig, ax = plt.subplots(figsize=(7.5, 6))
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', cbar=True, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap (Numerical Features)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=200)
    plt.close(fig)
    print("Saved: correlation_heatmap.png")

    # 4. Credit History vs Loan Approval
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cross_tab = pd.crosstab(df['Credit_History'], df['Loan_Status'])
    cross_tab.plot(kind='bar', stacked=False, ax=ax, color=['#e74c3c', '#2ecc71'])
    ax.set_title("Credit History vs Loan Approval", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Credit History (1: Good, 0: Poor)", fontsize=11)
    ax.set_ylabel("Number of Applicants", fontsize=11)
    ax.legend(["Not Approved (N)", "Approved (Y)"])
    plt.xticks(rotation=0)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "credit_history_vs_loan_approval.png"), dpi=200)
    plt.close(fig)
    print("Saved: credit_history_vs_loan_approval.png")

    # 5. Income vs Loan Approval
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.boxplot(x='Loan_Status', y='Applicant_Income', data=df, ax=ax, hue='Loan_Status', palette={'N': '#e74c3c', 'Y': '#2ecc71'}, legend=False)
    ax.set_title("Applicant Income vs Loan Approval", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Loan Status (Y: Approved, N: Not Approved)", fontsize=11)
    ax.set_ylabel("Applicant Income ($)", fontsize=11)
    # Clip view to 95th percentile for clarity while keeping outliers documented
    q95 = df['Applicant_Income'].quantile(0.95)
    ax.set_ylim(0, q95 * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "income_vs_loan_approval.png"), dpi=200)
    plt.close(fig)
    print("Saved: income_vs_loan_approval.png")

    print("All EDA plots generated successfully!")

if __name__ == "__main__":
    run_eda()
