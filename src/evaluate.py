from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(y_test, y_pred, kernel_name):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Report for {kernel_name} kernel:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.3f}\n")
    
    return accuracy
