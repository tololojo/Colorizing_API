import tensorflow as tf
import joblib

# Paths to the .keras and .h5 models
model_paths = {
    "HyperUnet_retrain_augmented_noise_corrected_Adam": "/home/tin/Projects/Colorizing_API/models/HyperUnet_retrain_augmented_noise_corrected_Adam_ckpt_epoch27_valloss0.0234.keras",
    "HyperUnet_retrain_no_augmented_corrected_Adam": "/home/tin/Projects/Colorizing_API/models/HyperUnet_retrain_no_augmented_corrected_Adam_ckpt_epoch27_valloss0.0230.keras",
    "base_model": "/home/tin/Projects/Colorizing_API/models/Hyper_U_Net.h5"  # Path to the baseline model
}

# Destination paths for the .joblib models
joblib_model_paths = {
    "HyperUnet_retrain_augmented_noise_corrected_Adam": "/home/tin/Projects/Colorizing_API/models_API/app/HyperUnet_retrain_augmented_noise_corrected_Adam.joblib",
    "HyperUnet_retrain_no_augmented_corrected_Adam": "/home/tin/Projects/Colorizing_API/models_API/app/HyperUnet_retrain_no_augmented_corrected_Adam.joblib",
    "base_model": "/home/tin/Projects/Colorizing_API/models_API/app/base_model.joblib"  # Path to save the baseline model
}

# Load each model and save it as a .joblib file
for model_name, model_path in model_paths.items():
    # Load the model (whether it's .keras or .h5)
    model = tf.keras.models.load_model(model_path)
    
    # Save the model as a .joblib file
    joblib.dump(model, joblib_model_paths[model_name])

    print(f"Saved {model_name} as .joblib at {joblib_model_paths[model_name]}")
