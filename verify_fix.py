
import sys
import os
import pickle
import pandas as pd
import mlflow

# Add the project root to sys.path so we can import flask_app.app
sys.path.append(os.getcwd())

try:
    print("Importing flask_app.app...")
    from flask_app import app
    print("Successfully imported flask_app.app")

    model = app.model
    vectorizer = app.vectorizer
    
    if model is None:
        print("Error: Model is None")
        sys.exit(1)
    
    if vectorizer is None:
        print("Error: Vectorizer is None")
        sys.exit(1)

    print(f"Model: {model}")
    print(f"Vectorizer: {vectorizer}")

    # Test compatibility
    text = "This is a test review to check feature compatibility."
    features = vectorizer.transform([text])
    print(f"Vectorizer produced {features.shape[1]} features.")
    
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
    
    print("Attempting prediction...")
    prediction = model.predict(features_df)
    print(f"Prediction successful: {prediction}")
    print("VERIFICATION PASSED: Model and Vectorizer are compatible.")

except ImportError as e:
    print(f"ImportError: {e}")
    # Fallback to manual verification logic if import fails due to dependencies
    sys.exit(1)
except Exception as e:
    print(f"Verification FAILED: {e}")
    sys.exit(1)
