from typing import List, Dict
import numpy as np


def get_global_average(items: List[Dict[str, float]]):
    """Accepts a list of dicts and returns the weighted global average of the data."""
    if not items:
        return 0  # Return 0 or suitable default if items list is empty
    
    counts = [item["count"] for item in items]
    total_count = sum(counts)
    if total_count == 0:
        return 0  # Avoid division by zero

    weighted_dp_fc = sum(np.array(item["dp_fc_mean"]) * item["count"] for item in items)
    global_dp_fc_mean = (weighted_dp_fc / total_count).tolist()
    
    global_average = {"dp_fc_mean": global_dp_fc_mean, "counts": counts}

    return global_average
