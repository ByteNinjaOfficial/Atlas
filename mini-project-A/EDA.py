import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def load_data(filepath):
    """Load the dataset from the specified filepath."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded dataset from {filepath}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def feature_engineering(df):
    """Create new features for deeper analysis before building ML models."""
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        # Simplify titles
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

def dataset_overview(df):
    """Print an overview of the dataset."""
    print("\n" + "="*40)
    print("1. Dataset Overview")
    print("="*40)
    print(f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"- Column names: {', '.join(df.columns.tolist())}")
    
    print("\n- Data types:")
    print(df.dtypes)
    
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n- Memory usage: {memory_usage:.2f} MB")
    
    print("\n- First 5 rows:")
    print(df.head())

def target_analysis(df):
    """Analyze the target variable."""
    print("\n" + "="*40)
    print("2. Target Analysis")
    print("="*40)
    
    if 'Survived' in df.columns:
        survived_count = df['Survived'].sum()
        total_count = len(df['Survived'].dropna())
        survival_rate = (survived_count / total_count) * 100
        death_rate = 100 - survival_rate
        
        print(f"Overall Survival Rate: {survival_rate:.2f}%")
        print(f"Deaths: {death_rate:.2f}%\n")
    else:
        print("'Survived' column not found.")

def missing_value_analysis(df, output_dir):
    """Analyze and visualize missing values with dynamic recommendations."""
    print("\n" + "="*40)
    print("3. Missing Value Analysis")
    print("="*40)
    
    missing_counts = df.isnull().sum()
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_counts,
        'Percentage (%)': missing_percentages
    })
    
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(by='Percentage (%)', ascending=False)
    
    if not missing_df.empty:
        print("\n- Missing Values Summary:")
        print(missing_df)
        
        # Dynamic suggestions
        print("\n- Suggestions for handling missing data:")
        for col in missing_df.index:
            pct = missing_df.loc[col, 'Percentage (%)']
            print(f"\nFeature: {col}")
            print(f"{pct:.1f}% missing")
            
            if pct > 50:
                print("Recommended: Drop the feature")
            elif pct > 5:
                if df[col].dtype == 'object':
                    print("Recommended: Mode Imputation or Treat as 'Unknown' Category")
                else:
                    print("Recommended: Median Imputation")
            else:
                print("Recommended: Drop rows with missing values or Impute")
    else:
        print("No missing values found.")

    # Generate and save missing value heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Value Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_value_heatmap.png'), dpi=300)
    plt.close()
    print(f"\n- Saved missing value heatmap to {output_dir}/missing_value_heatmap.png")

def statistical_summary(df):
    """Print strong statistical summaries including variance, skewness, and kurtosis."""
    print("\n" + "="*40)
    print("4. Statistical Summary")
    print("="*40)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(numerical_cols) > 0:
        print("\n- Numerical Statistics (Extended):")
        stats_df = df[numerical_cols].describe().T
        stats_df['variance'] = df[numerical_cols].var()
        stats_df['skewness'] = df[numerical_cols].skew()
        stats_df['kurtosis'] = df[numerical_cols].kurt()
        # Display key metrics clearly
        display_cols = ['mean', 'std', 'min', 'max', 'variance', 'skewness', 'kurtosis']
        print(stats_df[display_cols].to_string())
        
    if len(categorical_cols) > 0:
        print("\n- Categorical Statistics:")
        print(df[categorical_cols].describe(include=['object', 'category']))

def distribution_analysis(df, output_dir):
    """Perform comprehensive distribution analysis (Hist, KDE, Boxplot) for numeric features."""
    print("\n" + "="*40)
    print("5. Distribution Analysis")
    print("="*40)
    
    numeric_features = [col for col in ['Age', 'Fare'] if col in df.columns]
    
    for feature in numeric_features:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram
        sns.histplot(df[feature].dropna(), bins=30, ax=axes[0], color='skyblue')
        axes[0].set_title(f'{feature} Histogram')
        
        # KDE
        sns.kdeplot(df[feature].dropna(), ax=axes[1], color='coral', fill=True)
        axes[1].set_title(f'{feature} KDE')
        
        # Boxplot
        sns.boxplot(x=df[feature].dropna(), ax=axes[2], color='lightgreen')
        axes[2].set_title(f'{feature} Boxplot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'distribution_{feature.lower()}.png'), dpi=300)
        plt.close()
        print(f"- Saved combined distribution plots (Hist, KDE, Boxplot) for {feature}")

def univariate_analysis(df, output_dir):
    """Perform univariate analysis and save standard and interactive plots."""
    print("\n" + "="*40)
    print("6. Univariate Analysis")
    print("="*40)
    
    features = {
        'Survived': 'countplot',
        'Pclass': 'countplot',
        'Sex': 'countplot',
        'Embarked': 'countplot',
        'FamilySize': 'countplot'
    }
    
    for feature, plot_type in features.items():
        if feature not in df.columns:
            continue
            
        # Matplotlib version
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=feature, palette='Set2')
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'univariate_{feature.lower()}.png'), dpi=300)
        plt.close()
        
        # Interactive Plotly version for dashboard reuse
        fig = px.histogram(df, x=feature, title=f'Interactive Distribution of {feature}', 
                           color_discrete_sequence=['#636EFA'])
        fig.write_html(os.path.join(output_dir, f'plotly_univariate_{feature.lower()}.html'))
        
    print(f"- Saved univariate charts (PNG + Plotly HTML) to {output_dir}/")

def bivariate_analysis(df, output_dir):
    """Perform bivariate analysis using both Matplotlib and interactive Plotly charts."""
    print("\n" + "="*40)
    print("7. Bivariate Analysis")
    print("="*40)
    
    interactions = [
        ('Sex', 'Survived', 'countplot'),
        ('Pclass', 'Survived', 'countplot'),
        ('Embarked', 'Survived', 'countplot'),
        ('Age', 'Survived', 'histplot'),
        ('Fare', 'Survived', 'histplot')
    ]
    
    for x_col, hue_col, p_type in interactions:
        if x_col in df.columns and hue_col in df.columns:
            if p_type == 'countplot':
                # Matplotlib
                plt.figure(figsize=(8, 5))
                sns.countplot(data=df, x=x_col, hue=hue_col, palette='Set1')
                plt.title(f'{hue_col} by {x_col}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'bivariate_{x_col.lower()}_{hue_col.lower()}.png'), dpi=300)
                plt.close()
                
                # Plotly
                fig = px.histogram(df, x=x_col, color=hue_col, barmode='group', title=f'{hue_col} by {x_col}')
                fig.write_html(os.path.join(output_dir, f'plotly_bivariate_{x_col.lower()}_{hue_col.lower()}.html'))
                
            elif p_type == 'histplot':
                # Matplotlib
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=x_col, hue=hue_col, multiple='stack', kde=True, bins=30, palette='Set1')
                plt.title(f'{hue_col} by {x_col}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'bivariate_{x_col.lower()}_{hue_col.lower()}.png'), dpi=300)
                plt.close()
                
                # Plotly
                fig = px.histogram(df, x=x_col, color=hue_col, marginal='box', title=f'{hue_col} by {x_col}')
                fig.write_html(os.path.join(output_dir, f'plotly_bivariate_{x_col.lower()}_{hue_col.lower()}.html'))

    # Scatter
    if 'Age' in df.columns and 'Fare' in df.columns and 'Survived' in df.columns:
        fig = px.scatter(df, x='Age', y='Fare', color='Survived', title="Age vs Fare by Survival")
        fig.write_html(os.path.join(output_dir, 'plotly_scatter_age_fare.html'))

    print(f"- Saved bivariate charts (PNG + Plotly HTML) to {output_dir}/")

def correlation_analysis(df, output_dir):
    """Analyze correlations and feature relationships (Pairplot & Scatter Matrix)."""
    print("\n" + "="*40)
    print("8. Correlation & Relationship Analysis")
    print("="*40)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) < 2:
        print("Not enough numerical columns for correlation analysis.")
        return
        
    corr_matrix = df[numerical_cols].corr()
    
    print("\n- Correlation Matrix:")
    print(corr_matrix)
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    print(f"- Saved correlation heatmap to {output_dir}/correlation_heatmap.png")
    
    # Pairplot (Relationship Matrix)
    cols_for_pairplot = [col for col in ['Age', 'Fare', 'FamilySize', 'Survived'] if col in df.columns]
    if len(cols_for_pairplot) > 1:
        # We use dropna() specifically for the pairplot to avoid errors
        sns.pairplot(df[cols_for_pairplot].dropna(), 
                     hue='Survived' if 'Survived' in cols_for_pairplot else None, 
                     palette='Set1', diag_kind='kde')
        plt.savefig(os.path.join(output_dir, 'pairplot.png'), dpi=300)
        plt.close()
        print(f"- Saved pairplot to {output_dir}/pairplot.png")
    
    # Plotly Scatter Matrix
    if len(cols_for_pairplot) > 1:
        fig = px.scatter_matrix(df, 
                                dimensions=[c for c in cols_for_pairplot if c != 'Survived'], 
                                color='Survived' if 'Survived' in cols_for_pairplot else None, 
                                title="Interactive Scatter Matrix")
        fig.write_html(os.path.join(output_dir, 'plotly_scatter_matrix.html'))
        print(f"- Saved interactive scatter matrix to {output_dir}/plotly_scatter_matrix.html")

def calculate_survival_rate(df, column, value):
    """Helper to calculate survival rate for a specific category."""
    if 'Survived' not in df.columns:
        return 0.0
    subset = df[df[column] == value]
    if len(subset) == 0:
        return 0.0
    return subset['Survived'].mean() * 100

def feature_level_insights(df):
    """Compute and print dynamic, data-driven insights based on the dataset."""
    print("\n" + "="*40)
    print("9. Feature-Level & Data-Driven Insights")
    print("="*40)
    
    if 'Survived' not in df.columns:
        print("Target variable 'Survived' not found. Cannot compute insights.")
        return {}
        
    insights_dict = {}

    # Sex
    if 'Sex' in df.columns:
        print("\nFeature: Sex")
        female_rate = calculate_survival_rate(df, 'Sex', 'female')
        male_rate = calculate_survival_rate(df, 'Sex', 'male')
        print(f"- Female survival rate: {female_rate:.1f}%")
        print(f"- Male survival rate: {male_rate:.1f}%")
        insights_dict['Sex'] = f"Females survived at a much higher rate ({female_rate:.1f}%) compared to males ({male_rate:.1f}%)."

    # Pclass
    if 'Pclass' in df.columns:
        print("\nFeature: Passenger Class")
        class_rates = df.groupby('Pclass')['Survived'].mean() * 100
        for pclass, rate in class_rates.items():
            print(f"- Class {pclass} survival rate: {rate:.1f}%")
        insights_dict['Pclass'] = f"1st class had the highest survival rate ({class_rates.get(1, 0):.1f}%), indicating strong socioeconomic prioritization."

    # Embarked
    if 'Embarked' in df.columns:
        print("\nFeature: Embarked")
        emb_rates = df.groupby('Embarked')['Survived'].mean() * 100
        for emb, rate in emb_rates.items():
            print(f"- Port {emb} survival rate: {rate:.1f}%")
        insights_dict['Embarked'] = f"Passengers from Cherbourg (C) had a higher survival rate ({emb_rates.get('C', 0):.1f}%) compared to others."

    # Age Groups
    if 'Age' in df.columns:
        print("\nFeature: Age Groups")
        # Define age groups dynamically
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
        age_rates = df.groupby('AgeGroup', observed=False)['Survived'].mean() * 100
        for group, rate in age_rates.items():
            print(f"- {group} survival rate: {rate:.1f}%")
        insights_dict['Age'] = f"Children had the highest survival rate ({age_rates.get('Child', 0):.1f}%), while Seniors had the lowest ({age_rates.get('Senior', 0):.1f}%)."

    # Family Size
    if 'FamilySize' in df.columns:
        print("\nFeature: Family Size")
        fam_rates = df.groupby('FamilySize')['Survived'].mean() * 100
        best_size = fam_rates.idxmax() if not fam_rates.empty else "N/A"
        best_rate = fam_rates.max() if not fam_rates.empty else 0
        print(f"- Small families (2-4 members) usually survive better. Best size: {best_size} (Rate: {best_rate:.1f}%)")
        insights_dict['FamilySize'] = f"Passengers with a family size of {best_size} had the highest survival chances ({best_rate:.1f}%)."
        
    return insights_dict

def generate_summary_report(df, output_dir, insights_dict):
    """Generate a data-driven markdown summary report based on calculated metrics."""
    summary_path = os.path.join(output_dir, 'summary.md')
    
    try:
        with open(summary_path, 'w') as f:
            f.write("# Exploratory Data Analysis Summary - Titanic Dataset\n\n")
            
            f.write("## 1. Dataset Overview\n")
            f.write(f"- **Rows:** {df.shape[0]}\n")
            f.write(f"- **Columns:** {df.shape[1]}\n\n")
            
            if 'Survived' in df.columns:
                f.write("## 2. Target Analysis\n")
                surv_rate = df['Survived'].mean() * 100
                f.write(f"- **Overall Survival Rate:** {surv_rate:.2f}%\n")
                f.write(f"- **Overall Death Rate:** {100 - surv_rate:.2f}%\n\n")
            
            f.write("## 3. Statistical Highlights\n")
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in ['Age', 'Fare']:
                if col in num_cols:
                    f.write(f"- **{col}:** Mean = {df[col].mean():.2f}, Skewness = {df[col].skew():.2f}, Kurtosis = {df[col].kurt():.2f}\n")
            f.write("\n")
            
            f.write("## 4. Key Data-Driven Insights\n")
            for feature, insight in insights_dict.items():
                f.write(f"- **{feature}:** {insight}\n")
                
        print(f"\n- Generated dynamic summary report at {summary_path}")
    except Exception as e:
        print(f"\n- Error generating summary report: {e}")

def main():
    """Main execution function for the enhanced EDA pipeline."""
    print("Starting Enhanced Exploratory Data Analysis...")
    
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'Titanic-Dataset.csv')
    output_dir = os.path.join(script_dir, 'output')
    
    # Create output directory
    create_output_dir(output_dir)
    
    try:
        # Load data
        df = load_data(data_path)
        
        # 0. Feature Engineering
        df = feature_engineering(df)
        
        # 1. Dataset Overview
        dataset_overview(df)
        
        # 2. Target Analysis
        target_analysis(df)
        
        # 3. Missing Value Analysis
        missing_value_analysis(df, output_dir)
        
        # 4. Statistical Summary (Added Variance, Skew, Kurtosis)
        statistical_summary(df)
        
        # 5. Distribution Analysis (Hist, KDE, Boxplot)
        distribution_analysis(df, output_dir)
        
        # 6. Univariate Analysis (+ Plotly)
        univariate_analysis(df, output_dir)
        
        # 7. Bivariate Analysis (+ Plotly)
        bivariate_analysis(df, output_dir)
        
        # 8. Correlation Analysis (+ Pairplot/Scatter Matrix)
        correlation_analysis(df, output_dir)
        
        # 9. Dynamic Feature-Level Insights
        insights_dict = feature_level_insights(df)
        
        # 10. Data-Driven Summary Report
        generate_summary_report(df, output_dir, insights_dict)
        
        print("\n" + "="*40)
        print("Enhanced EDA completed successfully!")
        print(f"All standard and interactive Plotly outputs are saved in: {output_dir}")
        print("="*40)
        
    except Exception as e:
        logging.error(f"An error occurred during EDA: {e}")

if __name__ == "__main__":
    main()
