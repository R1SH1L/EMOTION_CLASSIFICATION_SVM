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

# Import your custom modules
from src.preprocessing import clean_text

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

# Call this after imports
download_nltk_data()

# Load models and vectorizer
@st.cache_resource
def load_models():
    try:
        models = {}
        for kernel in ['linear', 'rbf', 'poly']:
            model_path = f'models/svm_{kernel}.pkl'
            if os.path.exists(model_path):
                models[kernel] = joblib.load(model_path)
        
        vectorizer_path = 'models/vectorizer.pkl'
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            return models, vectorizer
        else:
            st.error("Vectorizer not found. Please train the model first.")
            return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
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
    
    # Load models
    models, vectorizer = load_models()
    
    if models is None or vectorizer is None:
        st.error("Models not loaded. Please ensure you have trained models in the 'models' directory.")
        st.info("Run the training script first: `python main.py`")
        return
    
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
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["Single Text", "Batch Upload"]
        )
        
        if input_method == "Single Text":
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
        
        else:  # Batch Upload
            uploaded_file = st.file_uploader(
                "Upload CSV file with 'text' column:",
                type=['csv']
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if 'text' not in df.columns:
                        st.error("CSV must contain a 'text' column")
                    else:
                        st.write("Preview of uploaded data:")
                        st.dataframe(df.head())
                        
                        if st.button("🔍 Predict Batch Emotions", type="primary"):
                            with st.spinner("Processing batch predictions..."):
                                predictions = []
                                
                                for text in df['text']:
                                    pred, _ = predict_emotion(
                                        str(text), 
                                        models[selected_kernel], 
                                        vectorizer
                                    )
                                    predictions.append(pred if pred else 'unknown')
                                
                                df['predicted_emotion'] = predictions
                                
                                # Display results
                                st.success("Batch prediction completed!")
                                st.dataframe(df)
                                
                                # Download results
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name="emotion_predictions.csv",
                                    mime="text/csv"
                                )
                                
                                # Show distribution
                                emotion_counts = pd.Series(predictions).value_counts()
                                fig = px.pie(
                                    values=emotion_counts.values,
                                    names=emotion_counts.index,
                                    title="Emotion Distribution in Batch"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error processing file: {e}")
    
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
                st.session_state.example_text = example_text
        
        # If example was clicked, show it in text area
        if 'example_text' in st.session_state:
            st.text_area(
                "Example Text:",
                value=st.session_state.example_text,
                height=100,
                disabled=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:**
    This emotion classification system uses Support Vector Machine (SVM) with different kernels 
    to predict emotions in text. The model was trained on emotion-labeled text data and can 
    classify text into 6 different emotional categories.
    """)

if __name__ == "__main__":
    main()