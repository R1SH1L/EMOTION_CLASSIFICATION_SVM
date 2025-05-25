import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Initial columns:", df.columns.tolist())
    print("Dataset shape:", df.shape)
    print("First few rows:")
    print(df.head())
    
    if len(df.columns) == 1 and ';' in df.columns[0]:
        print("Detected semicolon-separated data in single column. Fixing...")
        
        df = pd.read_csv(filepath, header=None, names=['raw_data'])
        
        split_data = df['raw_data'].str.split(';', n=1, expand=True)
        df['text'] = split_data[0]
        df['label'] = split_data[1]
        
        df = df.drop('raw_data', axis=1)
        
        df = df.dropna()
        
        print("Fixed data structure:")
        print("New columns:", df.columns.tolist())
        print("Dataset shape:", df.shape)
        print("First few rows:")
        print(df.head())
        print("Label distribution:")
        print(df['label'].value_counts())
    
    return df

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def preprocess_data(df):
    if 'text' not in df.columns:
        print("Available columns:", df.columns.tolist())
        raise ValueError("No 'text' column found after data loading.")
    
    print(f"Preprocessing text data...")
    df['text_clean'] = df['text'].apply(clean_text)
    
    df = df[df['text_clean'].str.len() > 0]
    
    print(f"After preprocessing: {len(df)} samples")
    return df