# CLIP Embedding API Documentation

This API provides unified access to OpenAI's CLIP model for generating embeddings from either text or images. The endpoint supports both modalities, enabling semantic similarity and multimodal search use cases.

## Endpoint URL

```
https://<databricks-instance>/serving-endpoints/<endpoint-name>/invocations
```

Replace `<databricks-instance>` and `<endpoint-name>` with your deployment details.

## Request Format

Send a POST request with the following JSON payload:

* **Text Embedding**
  ```json
  {
    "inputs": {"input_data": ["Your text here"]},
    "params": {"input_type": "text"}
  }
  ```
* **Image Embedding**
  ```json
  {
    "inputs": {"input_data": ["<base64-encoded-image>"]},
    "params": {"input_type": "image"}
  }
  ```
  * The image must be a base64-encoded string (no data:image prefix).

## Response Format

The response will be a JSON object containing a single embedding vector:

```json
{
  "predictions": [[0.123, -0.456, ...]]
}
```
* The vector is a list of floating-point numbers representing the CLIP embedding.

## Example Usage

### cURL Example (Text)
```bash
curl -X POST \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {"input_data": ["A photo of a cat"]},
    "params": {"input_type": "text"}
  }' \
  https://<databricks-instance>/serving-endpoints/<endpoint-name>/invocations
```

### cURL Example (Image)
```bash
curl -X POST \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {"input_data": ["<base64-encoded-image>"]},
    "params": {"input_type": "image"}
  }' \
  https://<databricks-instance>/serving-endpoints/<endpoint-name>/invocations
```

### Python Example (Text)
```python
import requests

url = "https://<databricks-instance>/serving-endpoints/<endpoint-name>/invocations"
headers = {"Authorization": "Bearer <token>", "Content-Type": "application/json"}
payload = {
    "inputs": {"input_data": ["A photo of a cat"]},
    "params": {"input_type": "text"}
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

### Python Example (Image)
```python
import requests
import base64

# Read and encode image
with open("cat.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

url = "https://<databricks-instance>/serving-endpoints/<endpoint-name>/invocations"
headers = {"Authorization": "Bearer <token>", "Content-Type": "application/json"}
payload = {
    "inputs": {"input_data": [img_b64]},
    "params": {"input_type": "image"}
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

## Notes
* For image inputs, provide only the base64-encoded string (no prefix like `data:image/jpeg;base64,`).
* The output is a single embedding vector per input.
* Ensure your token has permission to access the serving endpoint.