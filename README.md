# Emotion Classification using Support Vector Machine (SVM)

## Project Overview

This project implements a text classification system that automatically predicts emotions in text using Support Vector Machine (SVM) algorithms. The system compares different SVM kernel functions to determine the best classification performance for emotion detection.

## Project Structure

```
Emotion_Classification_SVM/
├── data/
│   ├── train.csv
│   └── kagle/
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   └── evaluate.py
├── models/
├── results/
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

## Features

- Text preprocessing and cleaning
- TF-IDF feature extraction
- SVM classification with multiple kernels (Linear, RBF, Polynomial)
- Model evaluation and comparison
- Streamlit web application interface
- Batch processing capabilities
- Model persistence

## Supported Emotions

The system classifies text into six emotional categories:
- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

## Installation

Create a virtual environment and install required packages:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main training script to train SVM models with different kernels:

```bash
python main.py
```

This will:
- Load and preprocess the training data
- Extract TF-IDF features
- Train SVM models with linear, RBF, and polynomial kernels
- Evaluate and compare model performance

### Saving Models for Deployment

```bash
python save_models.py
```

### Running the Web Application

```bash
streamlit run app.py
```

The web interface provides:
- Single text emotion prediction
- Batch file processing
- Kernel comparison
- Probability visualization
- Results download functionality

## Model Performance

Based on training results:

| Kernel | Accuracy | Best For |
|--------|----------|----------|
| Linear | 88% | Overall best performance |
| RBF | 84% | Complex patterns |
| Polynomial | 58% | Limited effectiveness |

## Data Format

Training data should be in CSV format with semicolon-separated values:
```
text;emotion_label
I am feeling happy today;joy
This makes me angry;anger
```

## Technical Details

### Preprocessing
- Text cleaning and normalization
- Removal of URLs, mentions, and special characters
- Stopword removal
- Tokenization using NLTK

### Feature Extraction
- TF-IDF vectorization
- N-gram support (unigrams and bigrams)
- Feature dimensionality optimization

### Model Training
- Support Vector Machine with multiple kernels
- Class imbalance handling
- Cross-validation support
- Hyperparameter optimization

## Dependencies

- Python 3.7+
- scikit-learn
- pandas
- numpy
- nltk
- streamlit
- plotly
- joblib

## File Descriptions

- `main.py`: Main training pipeline
- `app.py`: Streamlit web application
- `src/preprocessing.py`: Data loading and text preprocessing
- `src/feature_extraction.py`: TF-IDF feature extraction
- `src/train_model.py`: SVM model training
- `src/evaluate.py`: Model evaluation and metrics
- `save_models.py`: Model persistence for deployment

## Results

The trained models achieve competitive performance in emotion classification with the linear kernel showing the best overall results across all emotion categories.