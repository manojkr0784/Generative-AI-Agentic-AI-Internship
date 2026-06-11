import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ollama

# Function to Perform EDA and Generate Visualizations
def eda_analysis(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Fill missing values with mean for numerical columns
    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
        
    # Fill missing values with mode for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
        

    # Data Summary
    summary = df.describe(include='all').to_string()
    
    # Missigt values
    missing_values = df.isnull().sum().to_string()
    
    # Generate AI insights 
    insights = generate_ai_insights(summary)
    
    
    # Generate Data Visualizations
    
    plot_paths= generate_visualizations(df)
    
    return f"Data loaded Successfully! \n\n Data Summary:\n{summary}\n\nMissing Values:\n{missing_values}\n\nAI Insights:\n{insights}", plot_paths
   
    
# AI-Powered Insights Generation using Gemma( Ollama)
def generate_ai_insights(df_summary):
    prompt = f"Analyze the dataset summary and provide insights:\n{df_summary}"
    response = ollama.chat(model="gemma3:270m", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Function to Generate Data Visualizations
def generate_visualizations(df):
    plot_paths = []
    
   # Histogram for Numerical columns
    for col in df.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=30, kde=True, color='blue')
        plt.title(f'Distribution of {col}')
        path= f'{col}_distribution.png'
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()
        
        # Correlation Heatmap(only numerical columns)
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(8, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap')
        path = 'correlation_heatmap.png'
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()

    return plot_paths

# Gradio Interface
demo=gr.Interface(
    fn=eda_analysis,
    inputs=gr.File(type='filepath'),
    outputs=[gr.Textbox(label="EDA Report"), gr.Gallery(label="Visualizations")],
    title="EDA with LLM Insights",
    description="Upload a CSV file to perform Exploratory Data Analysis and get AI-generated insights."
)

# Launch the Gradio app
demo.launch(share=True)