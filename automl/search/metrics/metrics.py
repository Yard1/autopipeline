import numpy as np

# TODO make this better
def optimized_precision(accuracy, recall, specificity):
    """
    Ranawana, Romesh & Palade, Vasile. (2006). Optimized Precision - A New Measure for Classifier Performance Evaluation. 2254 - 2261. 10.1109/CEC.2006.1688586.
    """
    try:
        return accuracy - (np.abs(specificity - recall) / (specificity + recall))
    except Exception:
        return accuracy