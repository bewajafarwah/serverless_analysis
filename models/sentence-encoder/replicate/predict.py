from cog import BasePredictor, Input
from typing import List



MODEL_ID = "hfarwah/universal-sentence-encoder"
MODEL_CACHE = "model-cache"

import tensorflow as tf
import tensorflow_hub as hub

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.model = hub.load(MODEL_CACHE)

    def predict(self, 
                inputs: str= Input(
                    description="Input String",
                    default="Running on Replicate"
                )):
        tf_inputs = tf.constant([inputs])

        embeddings = self.model(tf_inputs)

        return embeddings

