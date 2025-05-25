import streamlit as st

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Emotion Classification with SVM",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import everything else
import joblib
import pandas as pd
import numpy as np
import nltk
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report
import json

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# Import your custom modules
from src.preprocessing import load_data, preprocess_data, clean_text
from src.feature_extraction import extract_features
from src.train_model import train_svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

@st.cache_resource
def train_and_save_models():
    """Train models if they don't exist"""
    
    # Check if models already exist
    models_exist = all([
        os.path.exists('models/svm_linear.pkl'),
        os.path.exists('models/svm_rbf.pkl'), 
        os.path.exists('models/svm_poly.pkl'),
        os.path.exists('models/vectorizer.pkl')
    ])
    
    if models_exist:
        st.success("✅ Pre-trained models found!")
        return True
    
    st.info("🔄 No pre-trained models found. Training models now...")
    
    try:
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Check if training data exists
        if not os.path.exists('data/train.csv'):
            st.error("❌ Training data not found. Please ensure 'data/train.csv' exists.")
            return False
        
        # Load and preprocess data
        with st.spinner("Loading and preprocessing data..."):
            df = load_data("data/train.csv")
            df = preprocess_data(df)
        
        # Extract features
        with st.spinner("Extracting features..."):
            X, vectorizer = extract_features(df['text_clean'])
            y = df['label']
        
        # Save vectorizer
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models for each kernel
        kernels = ['linear', 'rbf', 'poly']
        progress_bar = st.progress(0)
        
        for i, kernel in enumerate(kernels):
            with st.spinner(f"Training {kernel} kernel..."):
                # Use SVC with probability=True for better predictions
                model = SVC(
                    kernel=kernel, 
                    class_weight='balanced', 
                    probability=True,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Save model
                joblib.dump(model, f'models/svm_{kernel}.pkl')
                
                # Update progress
                progress_bar.progress((i + 1) / len(kernels))
        
        st.success("✅ Models trained and saved successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        return False

@st.cache_resource
def load_models():
    """Load models and vectorizer"""
    
    # First try to train models if they don't exist
    if not train_and_save_models():
        return None, None
    
    try:
        models = {}
        for kernel in ['linear', 'rbf', 'poly']:
            model_path = f'models/svm_{kernel}.pkl'
            if os.path.exists(model_path):
                models[kernel] = joblib.load(model_path)
            else:
                st.error(f"❌ Model not found: {model_path}")
                return None, None
        
        vectorizer_path = 'models/vectorizer.pkl'
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            return models, vectorizer
        else:
            st.error("❌ Vectorizer not found")
            return None, None
            
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

def predict_emotion(text, model, vectorizer):
    """Predict emotion for given text"""
    try:
        # Clean text
        cleaned_text = clean_text(text)
        
        # Transform text
        text_vector = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0] if hasattr(model, 'predict_proba') else None
        
        return prediction, probabilities
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

def main():
    # Title and description
    st.title("🎭 Emotion Classification with SVM")
    st.markdown("""
    This app uses Support Vector Machine (SVM) to classify emotions in text.
    Choose different SVM kernels to compare their performance on emotion detection.
    """)
    
    # Load models (this will train them if they don't exist)
    models, vectorizer = load_models()
    
    if models is None or vectorizer is None:
        st.error("❌ Failed to load or train models. Please check your data and try again.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("🔧 Model Configuration")
    available_kernels = list(models.keys())
    selected_kernel = st.sidebar.selectbox(
        "Select SVM Kernel:",
        available_kernels,
        index=0 if 'linear' in available_kernels else 0
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        user_text = st.text_area(
            "Enter your text:",
            placeholder="Type or paste your text here...",
            height=150
        )
        
        if st.button("🔍 Predict Emotion", type="primary"):
            if user_text.strip():
                with st.spinner("Analyzing emotion..."):
                    prediction, probabilities = predict_emotion(
                        user_text, 
                        models[selected_kernel], 
                        vectorizer
                    )
                    
                    if prediction:
                        # Display result
                        st.success(f"**Predicted Emotion: {prediction.upper()}**")
                        
                        # Show probabilities if available
                        if probabilities is not None:
                            prob_df = pd.DataFrame({
                                'Emotion': models[selected_kernel].classes_,
                                'Probability': probabilities
                            }).sort_values('Probability', ascending=False)
                            
                            fig = px.bar(
                                prob_df, 
                                x='Emotion', 
                                y='Probability',
                                title="Emotion Probabilities",
                                color='Probability',
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.header("📊 Model Information")
        
        # Model details
        st.info(f"""
        **Current Model:** SVM with {selected_kernel} kernel
        
        **Available Emotions:**
        - Joy 😊
        - Sadness 😢
        - Anger 😠
        - Fear 😨
        - Love ❤️
        - Surprise 😲
        """)
        
        # Example texts
        st.header("💡 Try These Examples")
        examples = {
            "Joy": "I'm so happy and excited about this amazing day!",
            "Sadness": "I feel so lonely and depressed today.",
            "Anger": "This is absolutely frustrating and makes me furious!",
            "Fear": "I'm terrified and scared about what might happen.",
            "Love": "I love you so much, you mean everything to me.",
            "Surprise": "Oh wow! I can't believe this unexpected news!"
        }
        
        for emotion, example_text in examples.items():
            if st.button(f"{emotion} Example", key=f"example_{emotion}"):
                st.text_area(
                    "Example Text:",
                    value=example_text,
                    height=100,
                    disabled=True,
                    key=f"example_display_{emotion}"
                )

if __name__ == "__main__":
    main()