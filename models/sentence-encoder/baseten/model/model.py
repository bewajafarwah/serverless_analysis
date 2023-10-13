"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""

import tensorflow as tf
from huggingface_hub import snapshot_download
import tensorflow_hub as hub


MODEL_ID = "hfarwah/universal-sentence-encoder"


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        self._data_dir = kwargs["data_dir"]
        self._model = None

    def load(self):
        model_path = snapshot_download(
            MODEL_ID,
            local_dir=self._data_dir
        )
        self.model = hub.load(model_path)


    def predict(self, request:dict):
        inputs = request.pop("inputs")
        tf_inputs = tf.constant([inputs])        

        embeddings = self.model(tf_inputs).numpy()

        return {"status": "success", "data": embeddings, "message": None}