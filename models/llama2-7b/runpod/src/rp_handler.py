''' infer.py for runpod worker '''

import os
import predict

import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schema import INPUT_SCHEMA

MODEL = predict.Predictor()
MODEL.setup()

def run(job):
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    answer = MODEL.predict(
        prompt=validated_input["prompt"],
        temperature=validated_input["temperature"],
        top_p=validated_input["top_p"],
        top_k=validated_input["top_k"],
        num_beams=validated_input["num_beams"],
        max_length=validated_input["max_length"],
    )

    return answer

runpod.serverless.start({"handler": run})