
import numpy as np
import tritonclient.http as httpclient
from tritonclient.http import InferInput, InferRequestedOutput

client = httpclient.InferenceServerClient(
    url="localhost:8000"
)

user_ids = np.array([["2"]], dtype=object)
movie_ids = np.array([["317"]], dtype=object)

user_input = InferInput("user_id", [1, 1], "BYTES")
movie_input = InferInput("movie_id", [1, 1], "BYTES")

user_input.set_data_from_numpy(user_ids)
movie_input.set_data_from_numpy(movie_ids)

output = InferRequestedOutput("dense_4")

response = client.infer(
    model_name="dcn_ranking",
    inputs=[user_input, movie_input],
    outputs=[output]
)

prediction = response.as_numpy("dense_4")

print("Predicted rating:", prediction)
print("TRITON INFERENCE TEST COMPLETED!")
