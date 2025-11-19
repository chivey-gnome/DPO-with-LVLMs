import json
from transformers import TrainerCallback

class DPOJSONLoggerCallback(TrainerCallback):
    """
    A callback to store every DPO training log entry into a JSON file.
    Logs are appended step-by-step for easy plotting later.
    """

    def __init__(self, json_path):
        self.json_path = json_path

        # Initialize the JSON file with an empty list
        with open(self.json_path, "w") as f:
            json.dump([], f)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called every time the trainer logs metrics.
        We append the entry into the JSON log file.
        """
        if logs is None:
            return

        # Make a copy & add global step number
        entry = logs.copy()
        entry["step"] = state.global_step

        # Append entry to the JSON array
        with open(self.json_path, "r+") as f:
            data = json.load(f)   # Load entire list
            data.append(entry)    # Append new log entry
            f.seek(0)             # Go back to beginning and overwrite
            json.dump(data, f, indent=2)
