# CLIP Model Serving Endpoints - API Guide

## Overview

CLIP text and image embedding models deployed on Databricks Model Serving for semantic search and similarity analysis.

**Endpoints:**
- `clip-text-embedding` - Text to 768-dimensional embeddings
- `clip-image-embedding` - Images to 768-dimensional embeddings

**Models:** `[your_catalog].[your_schema].clip_text_embedding` and `[your_catalog].[your_schema].clip_image_embedding`

---

## Authentication

```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()  # Uses default authentication (env vars, .databrickscfg, etc.)
```

---

## Text Embeddings

### Real-time Query

```python
import pandas as pd

# Single text embedding
input_df = pd.DataFrame({"input": ["A person working in an office"]})
response = w.serving_endpoints.query(
    name="clip-text-embedding",
    dataframe_records=input_df.to_dict('records')
)
embedding = response.predictions[0]  # 768-dimensional vector
```

### Batch Processing

```python
# Multiple texts at once
texts = [
    "Modern building architecture",
    "Technology equipment setup",
    "Urban landscape view"
]

input_df = pd.DataFrame({"input": texts})
response = w.serving_endpoints.query(
    name="clip-text-embedding",
    dataframe_records=input_df.to_dict('records')
)
embeddings = response.predictions  # List of 768-dimensional vectors
```

### SQL Usage

```sql
-- Single text embedding
SELECT ai_query('clip-text-embedding', request => 'A person in the office') AS embedding;

-- Batch from table
SELECT 
    content,
    ai_query('clip-text-embedding', request => content) AS embedding
FROM your_table
LIMIT 10;
```

---

## Image Embeddings

### Real-time Query

```python
import base64
from PIL import Image
from io import BytesIO

# Create test image and convert to base64
test_image = Image.new('RGB', (224, 224), color='blue')
buffer = BytesIO()
test_image.save(buffer, format='JPEG')
image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

input_df = pd.DataFrame({"input": [image_b64]})
response = w.serving_endpoints.query(
    name="clip-image-embedding",
    dataframe_records=input_df.to_dict('records')
)
embedding = response.predictions[0]  # 768-dimensional vector
```

### Load Image from File

```python
def load_image_as_base64(image_path):
    """Load image file and convert to base64 string for the model."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

image_b64 = load_image_as_base64('/path/to/image.jpg')
input_df = pd.DataFrame({"input": [image_b64]})
response = w.serving_endpoints.query(
    name="clip-image-embedding",
    dataframe_records=input_df.to_dict('records')
)
embedding = response.predictions[0]
```

### Batch Processing

```python
# Multiple images
image_paths = ['/path/to/img1.jpg', '/path/to/img2.jpg', '/path/to/img3.jpg']
image_b64_list = [load_image_as_base64(path) for path in image_paths]

input_df = pd.DataFrame({"input": image_b64_list})
response = w.serving_endpoints.query(
    name="clip-image-embedding",
    dataframe_records=input_df.to_dict('records')
)
embeddings = response.predictions  # List of 768-dimensional vectors
```

### SQL Usage

```sql
-- Single image embedding
SELECT ai_query('clip-image-embedding', request => image_base64) AS embedding
FROM your_image_table
LIMIT 1;

-- Batch from table
SELECT 
    image_id,
    ai_query('clip-image-embedding', request => image_base64) AS embedding
FROM [your_catalog].[your_schema].image_directory_embeddings
LIMIT 10;
```

---

## Local Model Loading

### Text Model

```python
import mlflow

# Load registered model from Unity Catalog
model = mlflow.pyfunc.load_model("models:/[your_catalog].[your_schema].clip_text_embedding/1")
embeddings = model.predict(["Sample text", "Another text"])
```

### Image Model

```python
# Load registered model from Unity Catalog
model = mlflow.pyfunc.load_model("models:/[your_catalog].[your_schema].clip_image_embedding/1")
embedding = model.predict([image_b64])  # Pass as list
```

### Spark UDF for Large-Scale Processing

```python
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType

# Load model
model = mlflow.pyfunc.load_model("models:/[your_catalog].[your_schema].clip_image_embedding/1")

def embedding_udf(image_base64):
    """Spark UDF to generate embeddings for each image."""
    if image_base64 is None:
        return None
    result = model.predict([image_base64])
    return result[0] if result else None

# Register and apply UDF
embedding_func = udf(embedding_udf, ArrayType(FloatType()))
df = spark.table("[your_catalog].[your_schema].image_directory")
embedded_df = df.withColumn("embeddings", embedding_func(col("image_base64")))

# Save results
embedded_df.write.format("delta").mode("overwrite").saveAsTable(
    "[your_catalog].[your_schema].image_directory_embeddings"
)
```

---

## Model Specifications

**Text Model:**
- Input: Text strings (max ~77 tokens)
- Output: 768-dimensional vectors
- Base: `openai/clip-vit-large-patch14`

**Image Model:**
- Input: Base64-encoded images (JPEG/PNG)
- Output: 768-dimensional vectors  
- Optimal size: 224x224 pixels
- Base: `openai/clip-vit-large-patch14`

---

## Configuration Examples

Replace the following placeholders with your actual values:

```python
# Example configuration
CATALOG_NAME = "your_catalog"
SCHEMA_NAME = "your_schema"
TEXT_MODEL_NAME = "clip_text_embedding"
IMAGE_MODEL_NAME = "clip_image_embedding"
TEXT_ENDPOINT_NAME = "clip-text-embedding"
IMAGE_ENDPOINT_NAME = "clip-image-embedding"
```

**Example from agriculture use case:**
```python
CATALOG_NAME = "autobricks"
SCHEMA_NAME = "agriculture"
TEXT_MODEL_NAME = "clip_text_embedding"
IMAGE_MODEL_NAME = "clip_image_embedding"
```

---

## Use Cases

These CLIP models can be used for:

- **Semantic Search:** Find images using natural language descriptions
- **Image Similarity:** Identify similar or duplicate images in your dataset
- **Content Classification:** Automatically categorize images based on content
- **Multi-modal Applications:** Bridge text and image understanding
- **Dataset Curation:** Clean and organize large image collections

---

## Performance Tips

- **Batch Processing:** Use batch queries for better throughput
- **Image Size:** Resize images to 224x224 for optimal performance
- **GPU Usage:** Ensure GPU endpoints are available for large-scale processing
- **Caching:** Consider caching frequently used embeddings
- **Parallel Processing:** Use Spark UDFs for distributed embedding generation

 