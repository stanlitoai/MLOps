from zenml import pipeline
from src.customerSatisfaction.steps.ingest_data import ingest_df
from src.customerSatisfaction.steps.clean_data import clean_df
from src.customerSatisfaction.steps.model_train import train_model
from src.customerSatisfaction.steps.evaluation import evaluate_model

@pipeline(enable_cache=True)
# @pipeline()
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    x_train, x_test, y_train, y_test = clean_df(df)
    model = train_model(x_train, x_test, y_train, y_test)
    mse, r2, rmse = evaluate_model(model, x_test, y_test)
    
    
    