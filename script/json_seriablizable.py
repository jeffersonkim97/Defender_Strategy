import numpy as np

def make_json_serializable(obj):
    """
    Recursively convert numpy arrays in obj to lists,
    so it can be JSON-serialized.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj