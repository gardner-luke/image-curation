# CLIP Embedding Endpoint Client

## Setup

```python
import os
import requests
import json
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
```

## Define the Endpoint Function

```python
def call_endpoint(input_data, input_type):
    """
    Call the CLIP embedding endpoint
    
    Args:
        input_data (str): Text string or base64 encoded image
        input_type (str): "text" or "image"
    """
    url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/clip-embedding-endpoint-356/invocations'
    headers = {
        'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "inputs": {"input_data": [input_data]},
        "params": {"input_type": input_type}
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f'Error {response.status_code}: {response.text}')
        return None
    
    return response.json()
```

## Test with Text

```python
# For text
result = call_endpoint("Hello world", "text")
print("Text result:", result)
```

## Test with Base64 Image

```python
# For base64 image
base64_image = "your_base64_encoded_image_here"
result = call_endpoint(base64_image, "image")
print("Image result:", result)
```

## Process Spark DataFrame

```python
# Create UDF for Spark
embedding_udf = udf(lambda img: call_endpoint(img, "image"), StringType())

# Apply to DataFrame
df_with_embeddings = df.withColumn("embeddings", embedding_udf(col("base64_image_column")))

# Show results
df_with_embeddings.select("base64_image_column", "embeddings").show(truncate=False)
```

## Alternative: Process DataFrame in Batches

```python
# Collect images and process in batches
images = df.select("base64_image_column").rdd.map(lambda row: row[0]).collect()

embeddings = []
for image in images:
    result = call_endpoint(image, "image")
    embeddings.append(result)
    
print(f"Processed {len(embeddings)} images")
```