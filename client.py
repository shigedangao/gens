import base64
import runpod
import os
import time

endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
payload = {
    "input": {
        "prompt": "Young tunisian women from la marsa, light brown long hair, with small lip and a small forehead looking at the camera, high quality, 8k, detailed, intricate, elegant with a smiling face and a flower on the left side of her head. Being in the center of tunis, sidi bou said in a sunny day eating a traditional tunisian dish called kefteji, wearing a modern casual outfit with a light blue dress with flower pattern, surrounded by beautiful white and blue buildings with bougainvillea flowers, vibrant colors, cinematic lighting, shallow depth of field, photo taken with an iPhone."
    }
}

runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint(endpoint_id=endpoint_id)

try:
    run_request = endpoint.run(payload)
    status = run_request.status()

    while status not in ["COMPLETED", "FAILED"]:
        print(f"Current status: {status}")
        status = run_request.status()
        # Wait before checking the status again
        time.sleep(5)

    if status == "FAILED":
        raise Exception("The request failed during processing.")
    
    result = run_request.output()
    b64value = result.get('image')
    image_data = base64.b64decode(b64value)

    with open("generated_image.png", "wb") as img_file:
        img_file.write(image_data)

except Exception as e:
    print(f"Error: {e}")
