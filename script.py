
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import joblib
import os
import json
from io import StringIO
import argparse


def model_fn(model_dir):
  
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

def input_fn(request_body, request_content_type):
    
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Desteklenmeyen content type: {request_content_type}")

def output_fn(prediction, content_type):

    if content_type == "application/json":
        return json.dumps({"prediction": prediction.tolist()})
    else:
        raise ValueError(f"Desteklenmeyen content type: {content_type}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--max_depth", type=int, default=None)
    
  
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    
    
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")
    
    args, _ = parser.parse_known_args()

   
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)
    
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

 
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        max_depth=args.max_depth
    )
    model.fit(X_train, y_train)

  
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    
    y_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

