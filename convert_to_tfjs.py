import tensorflow as tf
import os # For absolute paths and checks
import subprocess # To run tensorflowjs_converter
import shutil # For rmtree
from az_network import AlphaZeroNetwork 
from config import AZ_BEST_MODEL_FILE, TFJS_MODEL_DIR # TFJS_MODEL_DIR is in config
from env_const import NUM_ACTIONS # Import NUM_ACTIONS from env_const

# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths
# Ensure AZ_BEST_MODEL_FILE is just the filename, not a path already
ABS_AZ_BEST_MODEL_FILE = os.path.join(SCRIPT_DIR, AZ_BEST_MODEL_FILE)
# Change to a directory for SavedModel format
ABS_SAVED_MODEL_PATH = os.path.join(SCRIPT_DIR, 'switcharoo_az_saved_model')

# Ensure TFJS_MODEL_DIR from config is treated as relative to SCRIPT_DIR if it's a relative path
if TFJS_MODEL_DIR:
    if os.path.isabs(TFJS_MODEL_DIR):
        ABS_TFJS_OUTPUT_DIR = TFJS_MODEL_DIR
    else:
        ABS_TFJS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, TFJS_MODEL_DIR)
else:
    ABS_TFJS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'switcharoo_tfjs_model_default')


def export_model_for_tfjs(weights_path, output_saved_model_path):
    """
    Loads weights into the AlphaZeroNetwork and saves the full model in SavedModel format.
    Returns True on success, False on failure.
    """
    print(f"Attempting to use weights file: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights file not found at {weights_path}")
        return False

    print(f"Building model architecture...")
    az_net = AlphaZeroNetwork()

    print(f"Loading weights from: {weights_path}")
    try:
        az_net.load_model(weights_path)
    except Exception as e:
        print(f"ERROR: Failed to load weights from {weights_path}: {e}")
        return False

    print(f"Preparing to save full model in SavedModel format to: {output_saved_model_path}")
    try:
        # Ensure the directory is clean before saving
        if os.path.exists(output_saved_model_path):
            print(f"Output directory {output_saved_model_path} already exists. Removing it for a clean save.")
            shutil.rmtree(output_saved_model_path)
        os.makedirs(output_saved_model_path, exist_ok=True) # Ensure parent dirs exist, and dir itself
        
        print(f"Saving model using tf.saved_model.save()...")
        tf.saved_model.save(az_net.model, output_saved_model_path)
        print(f"Full model saved successfully in SavedModel format to {output_saved_model_path}")
        
        print(f"Contents of SavedModel directory ({output_saved_model_path}):")
        for item in os.listdir(output_saved_model_path):
            print(f"  - {item}")
        if os.path.exists(os.path.join(output_saved_model_path, 'variables')):
             print(f"Contents of variables subdirectory:")
             for item_var in os.listdir(os.path.join(output_saved_model_path, 'variables')):
                 print(f"    - {item_var}")

        return True
    except Exception as e:
        print(f"ERROR: Failed to save full model to {output_saved_model_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print(f"--- TensorFlow.js Model Conversion Script ---")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Using Keras weights file: {ABS_AZ_BEST_MODEL_FILE}")
    print(f"Intermediate SavedModel will be saved to: {ABS_SAVED_MODEL_PATH}")
    print(f"Final TensorFlow.js model output directory: {ABS_TFJS_OUTPUT_DIR}")
    
    if not os.path.exists(ABS_AZ_BEST_MODEL_FILE):
        print(f"CRITICAL ERROR: The specified best model weights file does not exist: {ABS_AZ_BEST_MODEL_FILE}")
        print("Please ensure training has been run and the file exists, or update AZ_BEST_MODEL_FILE in config.py.")
    else:
        # Pass the SavedModel path to the export function
        saved_model_export_success = export_model_for_tfjs(ABS_AZ_BEST_MODEL_FILE, ABS_SAVED_MODEL_PATH)
        
        if saved_model_export_success:
            print(f"\n--- Attempting TensorFlow.js Conversion from SavedModel ---")
            if not os.path.exists(ABS_TFJS_OUTPUT_DIR):
                try:
                    os.makedirs(ABS_TFJS_OUTPUT_DIR)
                    print(f"Created output directory: {ABS_TFJS_OUTPUT_DIR}")
                except Exception as e:
                    print(f"ERROR: Could not create output directory {ABS_TFJS_OUTPUT_DIR}: {e}")
                    saved_model_export_success = False

            if saved_model_export_success:
                command = [
                    "tensorflowjs_converter",
                    "--input_format", "tf_saved_model",
                    "--output_format", "tfjs_graph_model", # Explicitly graph model
                    "--quantize_float16", # Add 16-bit floating point quantization for mobile
                    ABS_SAVED_MODEL_PATH, 
                    ABS_TFJS_OUTPUT_DIR
                ]
                print(f"Executing command: {' '.join(command)}")
                
                try:
                    result = subprocess.run(command, capture_output=True, text=True, check=False)
                    
                    print("\n--- tensorflowjs_converter STDOUT ---")
                    print(result.stdout if result.stdout else "<No stdout>")
                    print("--- END STDOUT ---")

                    if result.stderr:
                        print("\n--- tensorflowjs_converter STDERR ---")
                        print(result.stderr)
                        print("--- END STDERR ---")
                    else:
                        print("<No stderr from tensorflowjs_converter>")

                    expected_model_json = os.path.join(ABS_TFJS_OUTPUT_DIR, "model.json")
                    if result.returncode == 0 and os.path.exists(expected_model_json):
                        print(f"\nConversion to TensorFlow.js format SUCCEEDED! (model.json found)")
                        print(f"Files should be in {ABS_TFJS_OUTPUT_DIR}")
                    elif result.returncode == 0:
                        print(f"\nWARNING: tensorflowjs_converter exited with code 0, but expected output file {expected_model_json} was NOT found.")
                        print("The conversion might have silently failed or produced unexpected output.")
                    else:
                        print(f"\nERROR: tensorflowjs_converter command failed with return code {result.returncode}.")
                        print("Please check the STDOUT/STDERR output above for details.")

                except FileNotFoundError:
                    print("ERROR: tensorflowjs_converter command not found. Please ensure it's installed and in your PATH.")
                    print("You might need to run: pip install tensorflowjs")
                except Exception as e:
                    print(f"An unexpected error occurred while trying to run tensorflowjs_converter: {e}")
        else:
            print("\nFailed to export Keras SavedModel. TensorFlow.js conversion cannot proceed.")

    print(f"--- Conversion Script Finished ---")
