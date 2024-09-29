from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
def save_cleaned_data(df):
    # Define the directory and file name for the cleaned data
    cleaned_file_path = 'processed/cleaned_data.csv'  # File saved in the 'static' folder

    # Save the cleaned DataFrame to a CSV file
    df.to_csv(cleaned_file_path, index=False)
    
    return cleaned_file_path

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Set folders for uploads and processed files
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Data cleaning functions
# def handle_missing_values(df):
#     return df.fillna(df.mean())
import pandas as pd

def handle_missing_values(df):
    # Select only numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    # Fill missing values in numerical columns with their mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def normalize_data(df):
    # Select only numeric columns for scaling
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Apply scaling only on numeric columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df


def handle_outliers(df):
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Filter out the rows with outliers
    df = df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return df

import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations(df):
    # Select only numeric columns for visualizations that require numeric data
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])

    # Plot histograms for numeric data
    if not numeric_cols.empty:
        numeric_cols.hist(bins=15, figsize=(15, 10), layout=(2, 3))
        plt.suptitle('Histograms of Numeric Features')
        plt.tight_layout()
        plt.savefig('processed/histograms.png')
        plt.close()

    # Generate scatter plots only for numeric columns (if more than one numeric column exists)
    if numeric_cols.shape[1] > 1:
        sns.pairplot(numeric_cols)
        plt.suptitle('Scatter Plots of Numeric Features')
        plt.savefig('processed/scatter_plots.png')
        plt.close()

    # Plot a correlation heatmap for numeric columns
    if numeric_cols.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig('processed/correlation_heatmap.png')
        plt.close()

    # Display the value counts for non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64'])
    for col in non_numeric_cols.columns:
        plt.figure(figsize=(8, 6))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Value Counts of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.savefig(f'processed/{col}_value_counts.png')
        plt.close()

    print("Visualizations generated and saved successfully.")


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        df = pd.read_csv(file)
        
        # Data Cleaning Steps
        df = handle_missing_values(df)
        df = remove_duplicates(df)
        df = normalize_data(df)  # Ensure normalization handles only numeric columns if needed
        df = handle_outliers(df)  # Use the updated handle_outliers function
        
        # Generate Visualizations with Correct Number of Arguments
        generate_visualizations(df)  # Only pass df, not any extra arguments
        
        # Save Cleaned Data
        cleaned_file_path = save_cleaned_data(df)
        
        flash('File processed and visualizations generated successfully!')
        return render_template('index.html', cleaned_file_path=cleaned_file_path)


# Create necessary folders if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
