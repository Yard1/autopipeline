def get_category_cardinalities(config, stage) -> dict:
    X = config.X.select_dtypes("category")
    return {col: X[col].nunique() for col in X.columns}
