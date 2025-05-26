if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(inputs, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    import mlflow
    import joblib

    # Unpack the vectorizer and model
    dv, model = inputs  

    with mlflow.start_run():
        # Log the model and vectorizer
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Save the DictVectorizer 
        joblib.dump(dv, 'dv.pkl')

        # Log the vectorizer file as an artifact in MLflow
        mlflow.log_artifact('dv.pkl', artifact_path='preprocessor')

        # Log additional params or metrics
        mlflow.log_param("model_type", "LinearRegression")

        print("Model registered with MLflow.")



