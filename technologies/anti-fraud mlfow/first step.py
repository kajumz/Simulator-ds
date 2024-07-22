import mlflow
import os

mlflow.set_tracking_uri(os.environ.get("TRACKING_URI"))
