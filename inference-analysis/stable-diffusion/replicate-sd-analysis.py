import time
import replicate
import json

prompt = "An astronaut riding a rainbow unicorn, cinematic, dramatic"


model = replicate.models.get("MODEL_NAME")
version = model.versions.get("VERSION_NUMBER")


TRIES = 50

latency = []

for i in range(TRIES):
    prediction = replicate.predictions.create(version=version, input={"prompt":prompt})

    start_time = time.time()
    while prediction.status != 'succeeded':
        prediction.reload()
    elapsed_time = time.time() - start_time

    latency.append(elapsed_time)

    print(elapsed_time)


with open("replicate-sd-results-2.json", "w") as f:
    json.dump({"replicate" : latency}, f)