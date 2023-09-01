import logging
import pandas as pd
from zenml import step
from customerSatisfaction.components.evl import MSE, R2Score, RMSE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker



# @step(experiment_tracker=experiment_tracker)
@step()
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame) -> Tuple[Annotated[float, "mse"],
                                                  Annotated[float, "r2_score"],
                                                  Annotated[float, "rmse"]]:
    
    try:
        # prediction = model.predict(x_test)
        # evaluation = Evaluation()
        # r2_score = evaluation.r2_score(y_test, prediction)
        # mlflow.log_metric("r2_score", r2_score)
        # mse = evaluation.mean_squared_error(y_test, prediction)
        # mlflow.log_metric("mse", mse)
        # rmse = np.sqrt(mse)
        # mlflow.log_metric("rmse", rmse)
        
        logging.info("Calulating model scores")
    
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)
        
        r2_class = R2Score()
        r2 = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2", r2)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        
        return mse, r2, rmse
        
    except Exception as e:
        logging.error("Error in evaluating model {}".format(e))
        raise e