import json
import requests
import time
import os

url = "HUGGING_FACE_ENDPOINT_URL"

HF_TOKEN = os.environ['HUGGING_FACE_TOKEN']

headers = {
  'Authorization': f'Bearer {HF_TOKEN}',
  'Content-Type': 'application/json'
}

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

TRIES = 20

for t in range(TRIES):
    print(t)
    for key, prompt in prompts.items():
        payload = json.dumps({
        "inputs": prompt,
        "parameters": {
            "temperature": 0.1,
            "top_p": 0.75,
            "top_k": 40,
            "repetition_penalty" : 1.1,
            "max_new_tokens": 512,
        }
        })

        start_time = time.time()
        response = requests.request("POST", url, headers=headers, data=payload)
        elapsed_time = time.time() - start_time

        if response.status_code != 200:
            print(response.status_code, response.content, t, key)        

        answer = json.loads(response.content)[0]['generated_text']

        results[key].append(elapsed_time)    
        answers[key].append(answer)

        print(key, elapsed_time)
    print()

with open("hf-llama-results.json", "w") as f:
    json.dump(results, f)


with open("hf-llama-answers.json", "w") as f:
    json.dump(answers, f)