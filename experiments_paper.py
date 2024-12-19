import subprocess

# Script and configuration paths
SCRIPT_PATH = "scripts/central_depth_estimation.py"

# Parameter grid
CONFIDENCES = [0.1, 0.3, 0.5, 0.7, 0.9]
MAX_DEPTHS = [20, 30]
METHODS = ["BBOX_BOTTOM", "BBOX_PERCENTILE", "SAM"]

# Function to run the script with given parameters
def run_depth_estimation(max_depth, confidence, method):
    print(f"Running with Max Depth={max_depth}, Confidence={confidence}, Method={method}")
  
    # Construct estimation overrides
    estimation_overrides = ','.join([
        f"\"general.max_depth={max_depth}\"",
        f"\"detection.bbox_confidence_threshold={confidence}\"",
        f"\"sampling.detection_sampling_method='{method}'\""
    ])
    
    estimation_overrides = "estimation.overrides=[" + estimation_overrides + "]"
    print(estimation_overrides)


    # Construct the full command
    command = [
        "python", SCRIPT_PATH,
        "override_present=True",
        estimation_overrides
    ]

    try:
        # Execute the command
        subprocess.run(command, check=True)
        print(f"Finished run with Max Depth={max_depth}, Confidence={confidence}, Method={method}")
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")

# Iterate over the parameter grid and execute the script
for max_depth in MAX_DEPTHS:
    for confidence in CONFIDENCES:
        for method in METHODS:
            run_depth_estimation(max_depth, confidence, method)
