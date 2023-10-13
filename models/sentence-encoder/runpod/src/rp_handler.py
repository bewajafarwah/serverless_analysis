''' infer.py for runpod worker '''

import os
import predict

import runpod
from runpod.serverless.utils.rp_validator import validate

from rp_schema import INPUT_SCHEMA


MODEL = predict.Predictor()
MODEL.setup()


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

  

    embeddings = MODEL.predict(
        inputs=validated_input["inputs"],
    )

    return embeddings


runpod.serverless.start({"handler": run})