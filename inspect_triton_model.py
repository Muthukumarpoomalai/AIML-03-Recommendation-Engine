
import tensorflow as tf

model_path = (
    "triton_model_repository/dcn_ranking/1/model.savedmodel"
)

model = tf.saved_model.load(model_path)

print("Available signatures:")
print(model.signatures.keys())

serving_fn = model.signatures["serving_default"]

print("\nINPUT SIGNATURE:")
print(serving_fn.structured_input_signature)

print("\nOUTPUT SIGNATURE:")
print(serving_fn.structured_outputs)
