from customerSatisfaction.pipelines.training_pipeline import train_pipeline
from zenml.client import Client


if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="data/olist_customers_dataset.csv")
    
    # mlflow ui --backend-store-uri "file:/home/anonymous/.config/zenml/local_stores/446a84ab-df09-4e5a-a4ed-190dae6e25b1/mlruns"
