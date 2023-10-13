import json
import requests
import time
import os 

url = "BASETEN_ENDPOINT_URL"

token = os.environ['BASETEN_TOKEN']

headers = {
  'Content-Type': 'application/json',
  'Authorization': f'Api-Key {token}'
}

prompt = "An astronaut riding a rainbow unicorn, cinematic, dramatic"

payload = json.dumps({
            "prompt": prompt
        })


TRIES = 1
latency = []

# import base64
# def save_image(content):
#   img=base64.b64decode(content)

#   img_file = open('image.jpeg', 'wb')
#   img_file.write(img)
#   img_file.close()

for i in range(TRIES):
    start_time = time.time()
    response = requests.request("POST", url=url, headers=headers, data=payload)
    elapsed_time = time.time() - start_time

    latency.append(elapsed_time)

    print(i, elapsed_time)

with open('baseten-sd-results-2.json', 'w') as f:
    json.dump({"baseten" : latency}, f)
