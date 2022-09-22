def get_category_cardinalities(config, stage) -> dict:
    X = config.X.select_dtypes("category")
    return {col: set(X[col].unique()) for col in X.columns}