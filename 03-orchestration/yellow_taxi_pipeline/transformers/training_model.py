if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression
    
    # Step 1: Select features and target
    features = ['PULocationID', 'DOLocationID']
    target = 'duration'

    # Step 2: Convert selected categorical features into dictionaries
    train_dicts = df[features].to_dict(orient='records')

    # Step 3: Vectorize the feature dictionaries
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    # Step 4: Extract the target values
    y_train = df[target].values

    # Step 5: Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 6: Print the intercept
    print(f"Intercept: {model.intercept_}")

    # Step 7: Return both the vectorizer and the model
    return dv, model


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
