import logging
import pandas as pd
from zenml import step
from customerSatisfaction.components.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training Label
        y_test: Training label
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        splitting_strategy = DataSplitStrategy()
        data_cleaning =DataCleaning(processed_data, splitting_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        return X_train, X_test, y_train, y_test 
        
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e