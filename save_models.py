from src.preprocessing import load_data, preprocess_data
from src.feature_extraction import extract_features
from src.train_model import train_svm
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.svm import SVC

os.makedirs('models', exist_ok=True)

df = load_data("data/train.csv")
df = preprocess_data(df)

X, vectorizer = extract_features(df['text_clean'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

joblib.dump(vectorizer, 'models/vectorizer.pkl')

kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    print(f"Training {kernel} kernel...")
    
    model = SVC(
        kernel=kernel, 
        class_weight='balanced', 
        probability=True, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    joblib.dump(model, f'models/svm_{kernel}.pkl')
    print(f"Saved: models/svm_{kernel}.pkl")

print("All models saved successfully!")