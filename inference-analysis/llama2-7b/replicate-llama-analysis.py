import time
import replicate
import json


model = replicate.models.get("model_name")
version = model.versions.get("version_number")


prompts = {
    "p1" : "What is the meaning of life?",
    "p2" :  "How would you explain Quantum mechains to a 5 year old?",
    "p3" : "In the hit 90s sitcom Friends, do you think Ross and Rachel were on a break?",
    "p4" :  "Can you draft an email to a fruit vendor asking for a quote of 2000 bananas?",
    "p5"  : "How far is the sun?"
}

results = {
    "p1" : [],
    "p2" : [],
    "p3" : [],
    "p4" : [],
    "p5" : []
}

answers = {
    "p1" : [],
    "p2" : [],
    "p3" : [],
    "p4" : [],
    "p5" : []
}

TRIES = 10

latency = []

for i in range(TRIES):
    print(i)
    for key, prompt in prompts.items():
        prediction = replicate.predictions.create(version=version, input={"prompt":prompt})

        start_time = time.time()
        while prediction.status != 'succeeded':
            prediction.reload()
        elapsed_time = time.time() - start_time

        answer = prediction.output

        latency.append(elapsed_time)

        print(key, elapsed_time)
        results[key].append(elapsed_time)    
        answers[key].append(answer)
    print()

with open("replicate-llama-results-2.json", "w") as f:
    json.dump(results, f)


with open("replicate-llama-answers-2.json", "w") as f:
    json.dump(answers, f)