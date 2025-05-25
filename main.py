from src.preprocessing import load_data, preprocess_data
from src.feature_extraction import extract_features
from src.train_model import train_svm
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

df = load_data("data/train.csv")
df = preprocess_data(df)

X, vectorizer = extract_features(df['text_clean'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for kernel in ['linear', 'rbf', 'poly']:
    model = train_svm(X_train, y_train, kernel=kernel)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, kernel)
