
import os
import tensorflow as tf

# Load trained DCN model
model = tf.keras.models.load_model(
    "models/dcn/dcn_ranking_model.keras"
)

# Create Triton model repository
output_path = "triton_model_repository/dcn_ranking/1/model.savedmodel"

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

# Export as TensorFlow SavedModel
model.save(
    output_path,
    save_format="tf"
)

print("DCN model loaded successfully!")
print("TensorFlow SavedModel exported!")
print("Triton model path:", output_path)
