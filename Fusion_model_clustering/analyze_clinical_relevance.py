#!/usr/bin/env python3
"""
Analyze clinical relevance of clusters by comparing cluster assignments
to clinical variables from the CHIMERA Task 3 dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
PROJECT_ROOT = Path(__file__).parent
CLUSTERS_PATH = PROJECT_ROOT / "analysis" / "clusters.csv"
CLINICAL_PATH = PROJECT_ROOT / "data" / "clinical_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "clinical_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load clusters and clinical data."""
    print("Loading data...")
    clusters_df = pd.read_csv(CLUSTERS_PATH)
    clinical_df = pd.read_csv(CLINICAL_PATH)
    
    # Merge - clinical data uses 'chimera_id_t3' column
    merged = clusters_df.merge(
        clinical_df,
        left_on='patient_id',
        right_on='chimera_id_t3',
        how='inner'
    )
    
    print(f"Loaded {len(merged)} patients with complete data")
    print(f"Cluster distribution:\n{merged['cluster'].value_counts().sort_index()}")
    print(f"\nClinical columns: {list(clinical_df.columns)}")
    
    return merged, clusters_df, clinical_df

def analyze_categorical_variables(data: pd.DataFrame, categorical_cols: List[str]):
    """Analyze categorical clinical variables across clusters."""
    results = {}
    
    for col in categorical_cols:
        if col not in data.columns or col in ['patient_id', 'cluster']:
            continue
            
        print(f"\nAnalyzing {col}...")
        crosstab = pd.crosstab(data['cluster'], data[col], margins=True)
        print(crosstab)
        
        # Chi-square test
        crosstab_no_margins = pd.crosstab(data['cluster'], data[col])
        if crosstab_no_margins.shape[0] > 1 and crosstab_no_margins.shape[1] > 1:
            chi2, p_value, dof, expected = stats.chi2_contingency(crosstab_no_margins)
            print(f"Chi-square test: chi2={chi2:.4f}, p-value={p_value:.4f}")
            results[col] = {'chi2': chi2, 'p_value': p_value, 'crosstab': crosstab}
        
        # Visualize
        plt.figure(figsize=(10, 6))
        crosstab_pct = pd.crosstab(data['cluster'], data[col], normalize='index') * 100
        crosstab_pct.plot(kind='bar', stacked=False)
        plt.title(f'Distribution of {col} across Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Percentage (%)')
        plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{col}_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    return results

def analyze_numeric_variables(data: pd.DataFrame, numeric_cols: List[str]):
    """Analyze numeric clinical variables across clusters."""
    results = {}
    
    for col in numeric_cols:
        if col not in data.columns or col in ['patient_id']:
            continue
            
        print(f"\nAnalyzing {col}...")
        
        # Summary statistics by cluster
        summary = data.groupby('cluster')[col].describe()
        print(summary)
        
        # ANOVA test
        clusters = sorted(data['cluster'].unique())
        cluster_data = [data[data['cluster'] == c][col].dropna() for c in clusters]
        
        if all(len(d) > 1 for d in cluster_data):
            f_stat, p_value = stats.f_oneway(*cluster_data)
            print(f"ANOVA: F={f_stat:.4f}, p-value={p_value:.4f}")
            results[col] = {'f_stat': f_stat, 'p_value': p_value, 'summary': summary}
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        data.boxplot(column=col, by='cluster', ax=axes[0])
        axes[0].set_title(f'{col} Distribution by Cluster')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel(col)
        
        # Violin plot
        cluster_list = []
        value_list = []
        for c in sorted(data['cluster'].unique()):
            values = data[data['cluster'] == c][col].dropna()
            cluster_list.extend([c] * len(values))
            value_list.extend(values)
        
        plot_df = pd.DataFrame({'cluster': cluster_list, col: value_list})
        sns.violinplot(data=plot_df, x='cluster', y=col, ax=axes[1])
        axes[1].set_title(f'{col} Distribution by Cluster (Violin Plot)')
        axes[1].set_xlabel('Cluster')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{col}_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    return results

def create_summary_visualization(data: pd.DataFrame, categorical_results: Dict, 
                                  numeric_results: Dict):
    """Create a summary visualization comparing clusters."""
    
    # Figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Cluster sizes
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_counts = data['cluster'].value_counts().sort_index()
    ax1.bar(cluster_counts.index.astype(str), cluster_counts.values, 
            color=['#E63946', '#457B9D', '#2A9D8F'])
    ax1.set_title('Cluster Sizes', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Number of Patients')
    
    # 2. Significant categorical variables (if any)
    ax2 = fig.add_subplot(gs[0, 1:])
    sig_cat = {k: v['p_value'] for k, v in categorical_results.items() 
               if v['p_value'] < 0.1}
    if sig_cat:
        cat_names = list(sig_cat.keys())
        p_vals = list(sig_cat.values())
        ax2.barh(cat_names, p_vals, color='orange')
        ax2.axvline(x=0.05, color='r', linestyle='--', label='p=0.05')
        ax2.set_title('Significant Categorical Associations (p < 0.1)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('P-value')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No significant\ncategorical associations\n(p < 0.1)', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('Categorical Associations', fontsize=12, fontweight='bold')
    
    # 3. Significant numeric variables (if any)
    ax3 = fig.add_subplot(gs[1, :])
    sig_num = {k: v['p_value'] for k, v in numeric_results.items() 
               if v['p_value'] < 0.1}
    if sig_num:
        num_names = list(sig_num.keys())
        p_vals = list(sig_num.values())
        ax3.barh(num_names, p_vals, color='teal')
        ax3.axvline(x=0.05, color='r', linestyle='--', label='p=0.05')
        ax3.set_title('Significant Numeric Associations (p < 0.1)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('P-value')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No significant\nnumeric associations\n(p < 0.1)', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Numeric Associations', fontsize=12, fontweight='bold')
    
    # 4. Cluster characteristics summary
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Create summary text
    summary_text = "CLUSTER CLINICAL RELEVANCE SUMMARY\n" + "="*60 + "\n\n"
    summary_text += f"Total Patients Analyzed: {len(data)}\n"
    summary_text += f"Clusters Identified: {len(data['cluster'].unique())}\n\n"
    
    summary_text += "Categorical Variables:\n"
    for col, res in categorical_results.items():
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        summary_text += f"  - {col}: p={res['p_value']:.4f} {sig}\n"
    
    summary_text += "\nNumeric Variables:\n"
    for col, res in numeric_results.items():
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        summary_text += f"  - {col}: p={res['p_value']:.4f} {sig}\n"
    
    summary_text += "\n(* p<0.05, ** p<0.01, *** p<0.001)\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(OUTPUT_DIR / "clinical_relevance_summary.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary visualization saved to {OUTPUT_DIR / 'clinical_relevance_summary.png'}")

def main():
    """Main analysis function."""
    print("="*60)
    print("CLUSTER CLINICAL RELEVANCE ANALYSIS")
    print("="*60)
    
    # Load data
    merged, clusters_df, clinical_df = load_data()
    
    # Identify column types
    categorical_cols = []
    numeric_cols = []
    
    for col in clinical_df.columns:
        if col in ['patient_id', 'chimera_id_t3']:
            continue
        if merged[col].dtype == 'object' or merged[col].dtype.name == 'category':
            categorical_cols.append(col)
        elif merged[col].dtype in ['int64', 'float64']:
            # Check if it's actually categorical (few unique values)
            if merged[col].nunique() < 10 and merged[col].nunique() < len(merged) * 0.1:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
    
    print(f"\nCategorical variables: {categorical_cols}")
    print(f"Numeric variables: {numeric_cols}")
    
    # Analyze
    categorical_results = analyze_categorical_variables(merged, categorical_cols)
    numeric_results = analyze_numeric_variables(merged, numeric_cols)
    
    # Create summary
    create_summary_visualization(merged, categorical_results, numeric_results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

