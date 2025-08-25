# Databricks notebook source
# MAGIC %md
# MAGIC # CLIP Model Serving with Unity Catalog
# MAGIC
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Create CLIP embedding models (image-only and image+text) using MLflow PyFunc
# MAGIC 2. Register models in Unity Catalog with proper versioning
# MAGIC 3. Create a unified model serving endpoint with named routes
# MAGIC 4. Test the deployed models with specific model selection
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - **Unity Catalog enabled workspace** with appropriate permissions
# MAGIC - **Databricks Runtime 13.2 ML or above** with dedicated access mode
# MAGIC - **Required Unity Catalog permissions**:
# MAGIC   - `USE CATALOG` and `USE SCHEMA` on target catalog/schema
# MAGIC   - `CREATE MODEL` permission on the schema
# MAGIC - **Cluster with GPU support** (recommended for CLIP model inference)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Install Dependencies
# MAGIC
# MAGIC **Note:** This cell will restart the Python kernel. User configurations will be set after the restart.

# COMMAND ----------

# Install required packages compatible with Databricks ML runtime
# Using specific versions to avoid conflicts
%pip install --upgrade \
    transformers \
    torch \
    torchvision \
    pillow \
    mlflow==2.11.0 \
    --quiet

# Restart Python to ensure packages are loaded correctly
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Configuration - UPDATE THESE VALUES
# MAGIC
# MAGIC **Please update the following variables before running this notebook:**

# COMMAND ----------

# ===== USER INPUT REQUIRED =====
# Update these values for your environment

# Unity Catalog configuration
CATALOG_NAME = "autobricks"  # e.g., "ml_models"
SCHEMA_NAME = "agriculture"    # e.g., "embedding_models" 
WORKSPACE_URL = "https://e2-demo-field-eng.cloud.databricks.com"  # e.g., "https://your-workspace.cloud.databricks.com"

# Model and endpoint configuration
MODEL_BASE_NAME = "clip_embedding_model_115"
ENDPOINT_NAME = "clip-embedding-unified-115"  # Single unified endpoint

# Data source configuration for testing (optional but recommended)
# If you have a table with real data, specify it here for testing
TEST_TABLE_NAME = None  # e.g., "your_catalog.your_schema.your_table" - Set to None to use synthetic data
IMAGE_COLUMN_NAME = "image_base64"  # Column name containing base64 encoded images
TEXT_COLUMN_NAME = "text_content"  # Column name containing text (for multimodal model)
TEST_BATCH_SIZE = 5  # Number of records to use for batch testing

# ===== END USER INPUT =====

# Validate required inputs
if CATALOG_NAME == "your_catalog_name" or SCHEMA_NAME == "your_schema_name":
    raise ValueError("Please update CATALOG_NAME and SCHEMA_NAME with your actual Unity Catalog names")

if WORKSPACE_URL == "your_workspace_url":
    raise ValueError("Please update WORKSPACE_URL with your actual Databricks workspace URL")

print("✅ Configuration validated successfully!")
print(f"📦 Target catalog: {CATALOG_NAME}")
print(f"📁 Target schema: {SCHEMA_NAME}")
print(f"🏢 Workspace URL: {WORKSPACE_URL}")
print(f"🚀 Unified endpoint: {ENDPOINT_NAME}")

if TEST_TABLE_NAME:
    print(f"📊 Test table: {TEST_TABLE_NAME}")
    print(f"   • Image column: {IMAGE_COLUMN_NAME}")
    print(f"   • Text column: {TEXT_COLUMN_NAME}")
else:
    print("📊 Using synthetic test data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Import Libraries and Setup

# COMMAND ----------

# Import required libraries
import mlflow
import torch
import pandas as pd
import numpy as np
import base64
import json
import requests
import time
from datetime import datetime
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from mlflow.models.signature import infer_signature
from mlflow.deployments import get_deploy_client

# Configure MLflow for Unity Catalog
mlflow.set_registry_uri("databricks-uc")

print("✅ Libraries imported successfully!")
print(f"🧠 MLflow version: {mlflow.__version__}")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"🤗 Transformers available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧪 Set MLflow Experiment

# COMMAND ----------

# Set experiment for tracking
experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/clip_embedding_experiments"
mlflow.set_experiment(experiment_name)

print(f"📊 MLflow experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Model Classes Definition
# MAGIC
# MAGIC We'll create two model classes:
# MAGIC 1. **CLIP_IMAGE_EMBEDDING**: Handles only image embeddings
# MAGIC 2. **CLIP_IMAGE_TEXT_EMBEDDING**: Handles both image and text embeddings
# MAGIC
# MAGIC Note: Using MLflow 2.11+ compatible signatures with `model_input` parameter

# COMMAND ----------

# Image-only embedding model
class CLIP_IMAGE_EMBEDDING(mlflow.pyfunc.PythonModel):
    """
    CLIP model for image embeddings only.
    Processes base64-encoded images and returns embedding vectors.
    Automatically detects column name or uses first column.
    """
    
    def load_context(self, context):
        """Load the CLIP model and processor."""
        from transformers import CLIPProcessor, CLIPModel
        
        # Initialize tokenizer and model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        print("✅ CLIP model loaded successfully for image embeddings")
    
    def _get_image_embedding_bytearray(self, base64_image_str):
        """Convert base64 image string to embedding vector."""
        import base64
        from PIL import Image
        from io import BytesIO
        
        # Decode base64 string back to bytearray
        decoded_bytearray = bytearray(base64.b64decode(base64_image_str))
        image = Image.open(BytesIO(decoded_bytearray))
        
        # Process image and get embeddings
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        
        return image_features.detach().numpy().tolist()[0]

    def predict(self, context, model_input, params=None):
        """Make predictions on a DataFrame containing base64 encoded images.
        Automatically detects column or uses first column.
        Using MLflow 2.11+ compatible signature."""
        
        # Auto-detect column: use single column or first column
        if len(model_input.columns) == 1:
            col_name = model_input.columns[0]
        else:
            # If multiple columns, allow params to specify, otherwise use first
            col_name = params.get('column_name', model_input.columns[0]) if params else model_input.columns[0]
        
        print(f"📍 Using column: '{col_name}' for image embeddings")
        
        # Process embeddings and return as list
        embeddings = model_input[col_name].apply(lambda x: self._get_image_embedding_bytearray(x))
        return embeddings.tolist()  # Return as list for proper format

# COMMAND ----------

# Image and text embedding model  
class CLIP_IMAGE_TEXT_EMBEDDING(mlflow.pyfunc.PythonModel):
    """
    CLIP model for both image and text embeddings.
    Can process either images (base64) or text based on input_type parameter.
    Requires column_name to be specified in params.
    """
    
    def load_context(self, context):
        """Load the CLIP model and processor."""
        from transformers import CLIPProcessor, CLIPModel
        
        # Initialize tokenizer and model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        print("✅ CLIP model loaded successfully for image and text embeddings")
    
    def _get_image_embedding_bytearray(self, base64_image_str):
        """Convert base64 image string to embedding vector."""
        from PIL import Image
        from io import BytesIO
        import base64
        
        # Decode base64 string back to bytearray
        decoded_bytearray = bytearray(base64.b64decode(base64_image_str))
        image = Image.open(BytesIO(decoded_bytearray))
        
        # Process image and get embeddings
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        
        return image_features.detach().numpy().tolist()[0]

    def _get_text_embedding(self, text):
        """Convert text to embedding vector."""
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**inputs)
        
        return text_features.detach().numpy().tolist()[0]

    def predict(self, context, model_input, params=None):
        """Make predictions based on input type (image or text).
        
        Required params:
        - input_type: 'image' or 'text'
        - column_name: name of the column containing the data
        
        For images: column should contain base64-encoded image strings
        For text: column should contain text strings
        """
        
        if params is None:
            raise ValueError("params must be provided with 'input_type' and 'column_name' keys")
            
        input_type = params.get('input_type')
        if not input_type:
            raise ValueError("'input_type' must be specified in params ('image' or 'text')")
        
        # Get column name - required for multimodal model
        col_name = params.get('column_name')
        if not col_name:
            # For single column DataFrames, auto-detect
            if len(model_input.columns) == 1:
                col_name = model_input.columns[0]
                print(f"📍 Auto-detected single column: '{col_name}'")
            else:
                raise ValueError(f"'column_name' must be specified in params when DataFrame has multiple columns. Available columns: {list(model_input.columns)}")
        
        # Validate column exists
        if col_name not in model_input.columns:
            raise ValueError(f"Column '{col_name}' not found in input data. Available columns: {list(model_input.columns)}")
        
        if input_type.lower() == 'image':
            print(f'🖼️ Processing image embeddings from column: {col_name}')
            embeddings = model_input[col_name].apply(lambda x: self._get_image_embedding_bytearray(x))
        elif input_type.lower() == 'text':
            print(f'📝 Processing text embeddings from column: {col_name}')
            embeddings = model_input[col_name].apply(lambda x: self._get_text_embedding(x))
        else:
            raise ValueError("input_type must be either 'image' or 'text'")
        
        # Return as list for proper format
        return embeddings.tolist()

print("✅ Model classes defined successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Create Sample Data and Load User Data

# COMMAND ----------

# Create synthetic sample data for testing
def create_sample_image_data():
    """Create a sample image for testing (simple colored square)."""
    from PIL import Image
    import base64
    from io import BytesIO
    
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

# Generate synthetic sample data
sample_image_base64 = create_sample_image_data()
sample_text = "a red square image"

# Create test DataFrames with synthetic data
synthetic_image_df = pd.DataFrame({'image_data': [sample_image_base64] * 10})  # 10 samples for batch testing
synthetic_text_df = pd.DataFrame({'text_data': [f"sample text {i}" for i in range(10)]})

print("✅ Synthetic sample data created")

# Load real data if table is specified
if TEST_TABLE_NAME:
    try:
        print(f"\n📂 Loading data from {TEST_TABLE_NAME}...")
        user_data_df = spark.table(TEST_TABLE_NAME).limit(TEST_BATCH_SIZE + 5).toPandas()
        
        # Check if specified columns exist
        if IMAGE_COLUMN_NAME in user_data_df.columns:
            image_test_df = user_data_df[[IMAGE_COLUMN_NAME]].dropna().head(TEST_BATCH_SIZE)
            print(f"✅ Loaded {len(image_test_df)} image records from column '{IMAGE_COLUMN_NAME}'")
        else:
            print(f"⚠️ Column '{IMAGE_COLUMN_NAME}' not found, using synthetic data")
            image_test_df = synthetic_image_df.head(TEST_BATCH_SIZE)
            
        if TEXT_COLUMN_NAME in user_data_df.columns:
            text_test_df = user_data_df[[TEXT_COLUMN_NAME]].dropna().head(TEST_BATCH_SIZE)
            print(f"✅ Loaded {len(text_test_df)} text records from column '{TEXT_COLUMN_NAME}'")
        else:
            print(f"⚠️ Column '{TEXT_COLUMN_NAME}' not found, using synthetic data")
            text_test_df = synthetic_text_df.head(TEST_BATCH_SIZE)
            
    except Exception as e:
        print(f"⚠️ Could not load table {TEST_TABLE_NAME}: {str(e)}")
        print("📊 Using synthetic test data instead")
        image_test_df = synthetic_image_df.head(TEST_BATCH_SIZE)
        text_test_df = synthetic_text_df.head(TEST_BATCH_SIZE)
else:
    print("📊 Using synthetic test data (no table specified)")
    image_test_df = synthetic_image_df.head(TEST_BATCH_SIZE)
    text_test_df = synthetic_text_df.head(TEST_BATCH_SIZE)

print(f"\n📁 Final test data shapes:")
print(f"   • Image data: {image_test_df.shape}")
print(f"   • Text data: {text_test_df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Model Registration - Version 1 (Image Only)
# MAGIC
# MAGIC We'll start by registering the image-only model as version 1.

# COMMAND ----------

# Register image-only model (Version 1)
with mlflow.start_run(run_name='clip_image_only_v1') as run:
    
    # Create and test the model
    image_model = CLIP_IMAGE_EMBEDDING()
    image_model.load_context(context=None)
    
    # Test prediction to create signature (using new signature)
    test_result = image_model.predict(context=None, model_input=image_test_df, params=None)
    
    # Infer signature
    signature = infer_signature(image_test_df, test_result)
    
    # Define pip requirements
    pip_requirements = [
        "mlflow==2.11.0",
        "torch",
        "torchvision", 
        "transformers",
        "accelerate",
        "pillow",
        "pandas"
    ]
    
    # Log the model
    model_info = mlflow.pyfunc.log_model(
        artifact_path="clip_image_only",
        python_model=image_model,
        registered_model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}",
        signature=signature,
        pip_requirements=pip_requirements,
        input_example=image_test_df,
        metadata={
            "model_type": "image_embedding",
            "base_model": "openai/clip-vit-large-patch14",
            "description": "CLIP model for image embeddings only",
            "version": "1.0",
            "capabilities": ["image_embedding"],
            "embedding_dimension": 768
        }
    )
    
    image_model_version_1_uri = model_info.model_uri
    image_run_id = run.info.run_id

print(f"✅ Image-only model registered successfully!")
print(f"📦 Model URI: {model_info.model_uri}")
print(f"🏃 Run ID: {image_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🆙 Model Registration - Version 2 (Image + Text)
# MAGIC
# MAGIC Now we'll register the enhanced model that handles both images and text as version 2.

# COMMAND ----------

# Register image+text model (Version 2)  
with mlflow.start_run(run_name='clip_image_text_v2') as run:
    
    # Create and test the model
    multimodal_model = CLIP_IMAGE_TEXT_EMBEDDING()
    multimodal_model.load_context(context=None)
    
    # Test both image and text predictions (using new signature)
    image_test_result = multimodal_model.predict(context=None, model_input=image_test_df, params={'input_type': 'image'})
    text_test_result = multimodal_model.predict(context=None, model_input=text_test_df, params={'input_type': 'text'})
    
    # Create a flexible signature that accepts any column name
    from mlflow.types.schema import Schema, ColSpec
    
    # Define a flexible input schema - using a generic column name
    # Create a sample DataFrame with a generic column for signature
    generic_df = pd.DataFrame({'input_data': [sample_image_base64]})
    
    # Infer signature with the generic DataFrame
    signature = infer_signature(
        generic_df, 
        [image_test_result[0]],  # Use first result as example output
        params={'input_type': 'image'}
    )
    
    # Define pip requirements
    pip_requirements = [
        "mlflow==2.11.0", 
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "pillow",
        "pandas"
    ]
    
    # Log the model
    model_info = mlflow.pyfunc.log_model(
        artifact_path="clip_image_text",
        python_model=multimodal_model,
        registered_model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}",
        signature=signature,
        pip_requirements=pip_requirements,
        input_example=generic_df,  # Use generic DataFrame as example
        metadata={
            "model_type": "multimodal_embedding",
            "base_model": "openai/clip-vit-large-patch14", 
            "description": "CLIP model for both image and text embeddings",
            "version": "2.0",
            "capabilities": ["image_embedding", "text_embedding"],
            "embedding_dimension": 768,
            "input_types": ["image", "text"],
            "notes": "Model accepts any column name via auto-detection or params['column_name']"
        }
    )
    
    multimodal_model_version_2_uri = model_info.model_uri
    multimodal_run_id = run.info.run_id

print(f"✅ Multimodal model registered successfully!")
print(f"📦 Model URI: {model_info.model_uri}")
print(f"🏃 Run ID: {multimodal_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏷️ Model Alias Management
# MAGIC
# MAGIC We'll set up model aliases for proper lifecycle management:
# MAGIC - **image-only**: Points to Version 1 (image-only model)
# MAGIC - **multimodal**: Points to Version 2 (image + text model)

# COMMAND ----------

# Initialize MLflow client for alias management
client = mlflow.MlflowClient()
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}"

# Set Version 1 alias
client.set_registered_model_alias(
    name=model_name,
    alias="image-only",
    version="1"
)

# Set Version 2 alias
client.set_registered_model_alias(
    name=model_name,
    alias="multimodal", 
    version="2"
)

# Update model descriptions
client.update_model_version(
    name=model_name,
    version="1",
    description="Image-only CLIP embedding model"
)

client.update_model_version(
    name=model_name,
    version="2", 
    description="Enhanced CLIP model with image and text capabilities"
)

print(f"✅ Model aliases configured successfully!")
print(f"🖼️ image-only: {model_name} version 1")
print(f"🎨 multimodal: {model_name} version 2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Create Unified Model Serving Endpoint
# MAGIC
# MAGIC We'll create a single endpoint that serves both models with named routes.
# MAGIC This allows you to specify which model to use when making requests.

# COMMAND ----------

# Initialize deployment client
deploy_client = get_deploy_client("databricks")

print("🔧 Creating unified model serving endpoint...")

# COMMAND ----------

# Create unified endpoint with both models
try:
    endpoint = deploy_client.create_endpoint(
        name=ENDPOINT_NAME,
        config={
            "served_entities": [
                {
                    "name": "image-only",  # Named route for image-only model
                    "entity_name": f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}",
                    "entity_version": "1",
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                    "workload_type": "GPU_SMALL"
                },
                {
                    "name": "multimodal",  # Named route for multimodal model
                    "entity_name": f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}",
                    "entity_version": "2",
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                    "workload_type": "GPU_SMALL"
                }
            ],
            "traffic_config": {
                "routes": [
                    {
                        "served_model_name": "image-only",
                        "traffic_percentage": 50
                    },
                    {
                        "served_model_name": "multimodal",
                        "traffic_percentage": 50
                    }
                ]
            }
        }
    )
    
    print(f"✅ Unified endpoint created: {ENDPOINT_NAME}")
    print(f"   • Route 'image-only': Version 1 (image embeddings only)")
    print(f"   • Route 'multimodal': Version 2 (image + text embeddings)")
    print(f"   • Traffic split: 50/50 (but you can specify model directly)")
    
except Exception as e:
    if "already exists" in str(e):
        print(f"ℹ️ Endpoint {ENDPOINT_NAME} already exists")
        print("   Attempting to update the endpoint configuration...")
        
        try:
            # Update existing endpoint
            deploy_client.update_endpoint(
                endpoint=ENDPOINT_NAME,
                config={
                    "served_entities": [
                        {
                            "name": "image-only",
                            "entity_name": f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}",
                            "entity_version": "1",
                            "workload_size": "Small",
                            "scale_to_zero_enabled": True,
                            "workload_type": "GPU_SMALL"
                        },
                        {
                            "name": "multimodal",
                            "entity_name": f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}",
                            "entity_version": "2",
                            "workload_size": "Small",
                            "scale_to_zero_enabled": True,
                            "workload_type": "GPU_SMALL"
                        }
                    ],
                    "traffic_config": {
                        "routes": [
                            {
                                "served_model_name": "image-only",
                                "traffic_percentage": 50
                            },
                            {
                                "served_model_name": "multimodal",
                                "traffic_percentage": 50
                            }
                        ]
                    }
                }
            )
            print(f"✅ Endpoint {ENDPOINT_NAME} updated successfully")
        except Exception as update_error:
            print(f"❌ Error updating endpoint: {str(update_error)}")
    else:
        print(f"❌ Error creating endpoint: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⏳ Wait for Endpoint to be Ready

# COMMAND ----------

def wait_for_endpoint_ready(endpoint_name, max_wait_minutes=15):
    """Wait for endpoint to be ready with status polling."""
    print(f"⏳ Waiting for endpoint '{endpoint_name}' to be ready...")
    
    max_wait_seconds = max_wait_minutes * 60
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait_seconds:
        try:
            endpoint_status = deploy_client.get_endpoint(endpoint=endpoint_name)
            state = endpoint_status.get('state', {})
            ready_state = state.get('ready', 'UNKNOWN')
            
            print(f"📊 Status: {ready_state}")
            
            if ready_state == 'READY':
                print(f"✅ Endpoint '{endpoint_name}' is ready!")
                return True
                
            elif ready_state in ['FAILED', 'ERROR']:
                print(f"❌ Endpoint '{endpoint_name}' failed to deploy")
                return False
                
            # Wait before next check
            time.sleep(30)
            
        except Exception as e:
            print(f"⚠️ Error checking endpoint status: {str(e)}")
            time.sleep(30)
    
    print(f"⏰ Timeout waiting for endpoint '{endpoint_name}' to be ready")
    return False

# Wait for endpoint
print("🚀 Waiting for endpoint to be ready...")
endpoint_ready = wait_for_endpoint_ready(ENDPOINT_NAME)

if endpoint_ready:
    print("🎉 Endpoint is ready for testing!")
else:
    print("⚠️ Endpoint may not be ready. Check the Databricks UI for details.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧪 Test Unified Endpoint with Named Routes
# MAGIC
# MAGIC We'll test both models with:
# MAGIC 1. **Single record requests** - One input at a time
# MAGIC 2. **Batch requests** - Multiple inputs in one request (5 records)

# COMMAND ----------

# Get authentication token
notebook_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Helper function to display test results
def display_test_results(response, test_type, model_name):
    """Helper to display test results in a formatted way."""
    if response.status_code == 200:
        result = response.json()
        embeddings = result.get('predictions', [])
        if embeddings and len(embeddings) > 0:
            print(f"✅ {model_name} {test_type} test successful!")
            print(f"   • Records processed: {len(embeddings)}")
            # Check if embedding is nested or flat
            if isinstance(embeddings[0], list):
                print(f"   • Embedding dimension: {len(embeddings[0])}")
            else:
                print(f"   • Embedding dimension: 1 (Warning: may be incorrectly formatted)")
            return True
        else:
            print(f"⚠️ No embeddings returned for {model_name} {test_type}")
            return False
    else:
        print(f"❌ {model_name} {test_type} test failed: {response.status_code}")
        error_msg = response.text[:300] if len(response.text) > 300 else response.text
        print(f"   • Error: {error_msg}...")
        return False

# COMMAND ----------

# Test image-only model
def test_image_only_model():
    """Test the image-only model with single and batch requests."""
    print("🧪 Testing Image-Only Model")
    print("=" * 50)
    
    endpoint_url = f"{WORKSPACE_URL}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {notebook_token}"
    }
    
    # Get column name from test dataframe
    col_name = image_test_df.columns[0]
    
    # 1. Single Record Test
    print("\n📍 Single Record Test:")
    single_record = image_test_df.iloc[0:1].to_dict('records')[0]
    single_test_data = {
        "dataframe_records": [single_record],
        "served_model_name": "image-only"
    }
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=single_test_data)
        single_success = display_test_results(response, "single", "Image-only")
    except Exception as e:
        print(f"❌ Error in single test: {str(e)}")
        single_success = False
    
    # 2. Batch Test (5 records)
    print("\n📍 Batch Test (5 records):")
    batch_records = image_test_df.head(TEST_BATCH_SIZE).to_dict('records')
    batch_test_data = {
        "dataframe_records": batch_records,
        "served_model_name": "image-only"
    }
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=batch_test_data)
        batch_success = display_test_results(response, "batch", "Image-only")
    except Exception as e:
        print(f"❌ Error in batch test: {str(e)}")
        batch_success = False
    
    print("\n" + "=" * 50)
    return single_success and batch_success

# COMMAND ----------

# Test multimodal model
def test_multimodal_model():
    """Test the multimodal model with both image and text, single and batch."""
    print("🧪 Testing Multimodal Model")
    print("=" * 50)
    
    endpoint_url = f"{WORKSPACE_URL}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {notebook_token}"
    }
    
    results = []
    
    # Test with IMAGE data
    print("\n🖼️ IMAGE EMBEDDINGS:")
    image_col_name = image_test_df.columns[0]
    
    # 1. Single Image Test
    print("\n📍 Single Record Test:")
    single_image_record = image_test_df.iloc[0:1].to_dict('records')[0]
    single_image_data = {
        "dataframe_records": [single_image_record],
        "served_model_name": "multimodal",
        "params": {
            "input_type": "image",
            "column_name": image_col_name  # Specify column if needed
        }
    }
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=single_image_data)
        results.append(display_test_results(response, "single image", "Multimodal"))
    except Exception as e:
        print(f"❌ Error in single image test: {str(e)}")
        results.append(False)
    
    # 2. Batch Image Test
    print("\n📍 Batch Test (5 records):")
    batch_image_records = image_test_df.head(TEST_BATCH_SIZE).to_dict('records')
    batch_image_data = {
        "dataframe_records": batch_image_records,
        "served_model_name": "multimodal",
        "params": {
            "input_type": "image",
            "column_name": image_col_name
        }
    }
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=batch_image_data)
        results.append(display_test_results(response, "batch image", "Multimodal"))
    except Exception as e:
        print(f"❌ Error in batch image test: {str(e)}")
        results.append(False)
    
    # Test with TEXT data
    print("\n📝 TEXT EMBEDDINGS:")
    text_col_name = text_test_df.columns[0]
    
    # 3. Single Text Test
    print("\n📍 Single Record Test:")
    single_text_record = text_test_df.iloc[0:1].to_dict('records')[0]
    single_text_data = {
        "dataframe_records": [single_text_record],
        "served_model_name": "multimodal",
        "params": {
            "input_type": "text",
            "column_name": text_col_name
        }
    }
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=single_text_data)
        results.append(display_test_results(response, "single text", "Multimodal"))
    except Exception as e:
        print(f"❌ Error in single text test: {str(e)}")
        results.append(False)
    
    # 4. Batch Text Test
    print("\n📍 Batch Test (5 records):")
    batch_text_records = text_test_df.head(TEST_BATCH_SIZE).to_dict('records')
    batch_text_data = {
        "dataframe_records": batch_text_records,
        "served_model_name": "multimodal",
        "params": {
            "input_type": "text",
            "column_name": text_col_name
        }
    }
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=batch_text_data)
        results.append(display_test_results(response, "batch text", "Multimodal"))
    except Exception as e:
        print(f"❌ Error in batch text test: {str(e)}")
        results.append(False)
    
    print("\n" + "=" * 50)
    return all(results)

# COMMAND ----------

# Run all tests
if endpoint_ready:
    print("🚀 Running Comprehensive Endpoint Tests\n")
    
    # Test image-only model
    image_only_success = test_image_only_model()
    
    print()  # Add spacing
    
    # Test multimodal model
    multimodal_success = test_multimodal_model()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Image-only model: {'PASSED' if image_only_success else 'FAILED'}")
    print(f"✅ Multimodal model: {'PASSED' if multimodal_success else 'FAILED'}")
    
    if image_only_success and multimodal_success:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")
else:
    print("⚠️ Skipping tests - endpoint not ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 How to Call the Endpoint from Your Application
# MAGIC
# MAGIC The multimodal model requires you to specify:
# MAGIC - `input_type`: either 'image' or 'text'
# MAGIC - `column_name`: the name of your column containing the data (auto-detected for single-column DataFrames)
# MAGIC
# MAGIC For images, the column must contain base64-encoded image strings.

# COMMAND ----------

# Example code for calling from external applications
print("""
📚 ENDPOINT USAGE EXAMPLES:

================================================================================
1️⃣ SINGLE RECORD REQUESTS
================================================================================

🖼️ Image-Only Model (Single Record):
------------------------------------
import requests
import json

endpoint_url = f"{WORKSPACE_URL}/serving-endpoints/{ENDPOINT_NAME}/invocations"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_TOKEN"
}

# Single image embedding - column name auto-detected for single column
data = {
    "dataframe_records": [
        {"my_image_column": "BASE64_ENCODED_IMAGE_STRING"}
    ],
    "served_model_name": "image-only"
}

response = requests.post(endpoint_url, headers=headers, json=data)
embedding = response.json()['predictions'][0]  # Single 768-dim embedding vector

📝 Multimodal Text (Single Record):
----------------------------------
# You specify the column name explicitly
data = {
    "dataframe_records": [
        {"product_description": "analyze this text"}
    ],
    "served_model_name": "multimodal",
    "params": {
        "input_type": "text",
        "column_name": "product_description"  # Specify your column name
    }
}

response = requests.post(endpoint_url, headers=headers, json=data)
embedding = response.json()['predictions'][0]

================================================================================
2️⃣ BATCH REQUESTS (Multiple Records)
================================================================================

🖼️ Batch Image Processing:
--------------------------
# Process multiple images with your custom column name
batch_data = {
    "dataframe_records": [
        {"photo_base64": "BASE64_IMAGE_1"},
        {"photo_base64": "BASE64_IMAGE_2"},
        {"photo_base64": "BASE64_IMAGE_3"},
        {"photo_base64": "BASE64_IMAGE_4"},
        {"photo_base64": "BASE64_IMAGE_5"}
    ],
    "served_model_name": "multimodal",
    "params": {
        "input_type": "image",
        "column_name": "photo_base64"  # Your column name
    }
}

response = requests.post(endpoint_url, headers=headers, json=batch_data)
embeddings = response.json()['predictions']  # List of 5 embedding vectors (768-dim each)

📝 Batch Text Processing:
------------------------
# Process multiple text records with custom column
batch_data = {
    "dataframe_records": [
        {"review_text": "first document", "other_field": "ignored"},
        {"review_text": "second document", "other_field": "ignored"},
        {"review_text": "third document", "other_field": "ignored"}
    ],
    "served_model_name": "multimodal",
    "params": {
        "input_type": "text",
        "column_name": "review_text"  # Specify which column to process
    }
}

response = requests.post(endpoint_url, headers=headers, json=batch_data)
embeddings = response.json()['predictions']  # List of text embedding vectors

================================================================================
💡 KEY POINTS
================================================================================

1. **Column Names**: Use any column name you want - just specify it in params
2. **Image Format**: Images must be base64-encoded strings
3. **Single Column**: If DataFrame has only 1 column, column_name is optional
4. **Batch Size**: Process 5-50 records per request for optimal performance
5. **Error Handling**: Model will tell you available columns if you specify wrong one

================================================================================
🔧 PANDAS INTEGRATION EXAMPLE
================================================================================

import pandas as pd
import base64

# Your DataFrame with custom column names
df = pd.DataFrame({
    'item_image_b64': [...],  # Your base64 images
    'item_description': [...],  # Your text data
    'item_id': [...]          # Other columns (ignored by model)
})

# Get image embeddings
image_request = {
    "dataframe_records": df[['item_image_b64']].to_dict('records'),
    "served_model_name": "multimodal",
    "params": {
        "input_type": "image",
        "column_name": "item_image_b64"
    }
}

# Get text embeddings  
text_request = {
    "dataframe_records": df[['item_description']].to_dict('records'),
    "served_model_name": "multimodal",
    "params": {
        "input_type": "text",
        "column_name": "item_description"
    }
}

""")

print(f"\n🔗 Endpoint URL: {WORKSPACE_URL}/serving-endpoints/{ENDPOINT_NAME}/invocations")
print(f"🔑 Remember to replace 'YOUR_TOKEN' with your actual authentication token")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Summary and Next Steps

# COMMAND ----------

# Display final status
print("="*60)
print("🎉 MODEL SERVING SETUP COMPLETE!")
print("="*60)

print(f"\n📦 Model Registry:")
print(f"   • Catalog: {CATALOG_NAME}")
print(f"   • Schema: {SCHEMA_NAME}")  
print(f"   • Model: {MODEL_BASE_NAME}")

print(f"\n🚀 Unified Serving Endpoint:")
print(f"   • Name: {ENDPOINT_NAME}")
print(f"   • Status: {'Ready ✅' if endpoint_ready else 'Not Ready ⚠️'}")

print(f"\n🔗 Available Model Routes:")
print(f"   • 'image-only': Version 1 (Image embeddings only)")
print(f"   • 'multimodal': Version 2 (Image + Text embeddings)")

print(f"\n🏷️ Model Aliases:")
try:
    aliases = client.get_model_version_by_alias(name=model_name, alias="image-only")
    print(f"   • image-only: Version {aliases.version}")
except:
    print(f"   • image-only: Version 1")

try:
    aliases = client.get_model_version_by_alias(name=model_name, alias="multimodal")
    print(f"   • multimodal: Version {aliases.version}")
except:
    print(f"   • multimodal: Version 2")

print(f"\n📋 Next Steps:")
print(f"   1. Monitor endpoint performance in Databricks UI")
print(f"   2. Set up inference tables for monitoring (optional)")
print(f"   3. Configure endpoint permissions as needed")
print(f"   4. Test endpoint with your actual data")
print(f"   5. Set up automated retraining pipelines")

print(f"\n🔍 Useful URLs:")
print(f"   • Serving UI: {WORKSPACE_URL}/#mlflow/endpoints")
print(f"   • Models: {WORKSPACE_URL}/#mlflow/models")
print(f"   • Experiments: {WORKSPACE_URL}/#mlflow/experiments")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧹 Optional: Cleanup Resources
# MAGIC
# MAGIC Uncomment and run the following cell to clean up endpoints and models when no longer needed.

# COMMAND ----------

# # UNCOMMENT TO ENABLE CLEANUP - BE CAREFUL!
# # This will delete your endpoints and models

# def cleanup_resources():
#     """Clean up serving endpoints and models."""
#     print("🧹 Starting cleanup process...")
    
#     # Delete serving endpoint
#     try:
#         deploy_client.delete_endpoint(endpoint=ENDPOINT_NAME)
#         print(f"🗑️ Deleted endpoint: {ENDPOINT_NAME}")
#     except Exception as e:
#         print(f"⚠️ Could not delete endpoint: {str(e)}")
    
#     # Delete model versions (optional - be very careful!)
#     # try:
#     #     client.delete_model_version(name=model_name, version="1")
#     #     client.delete_model_version(name=model_name, version="2")
#     #     client.delete_registered_model(name=model_name)
#     #     print(f"🗑️ Deleted model: {model_name}")
#     # except Exception as e:
#     #     print(f"⚠️ Could not delete model: {str(e)}")
    
#     print("✅ Cleanup completed!")

# # Uncomment the line below to run cleanup
# # cleanup_resources()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Additional Resources and Documentation
# MAGIC
# MAGIC ### Key Concepts Covered
# MAGIC
# MAGIC 1. **Unity Catalog Integration**
# MAGIC    - Model registration with three-level namespace (`catalog.schema.model`)
# MAGIC    - Proper permissions and access control
# MAGIC    - Model versioning and lifecycle management
# MAGIC
# MAGIC 2. **MLflow Model Serving**
# MAGIC    - PyFunc model wrapper for custom models
# MAGIC    - Model signatures for input/output validation (MLflow 2.11+ compatible)
# MAGIC    - Deployment using MLflow Deployments SDK
# MAGIC
# MAGIC 3. **Unified Endpoint Architecture**
# MAGIC    - Single endpoint serving multiple model versions
# MAGIC    - Named routes for specific model selection
# MAGIC    - Simplified deployment and management
# MAGIC
# MAGIC 4. **CLIP Model Implementation**
# MAGIC    - Image embedding generation from base64 images
# MAGIC    - Text embedding generation
# MAGIC    - Multimodal model serving with parameter-based routing
# MAGIC
# MAGIC ### Databricks Documentation Links
# MAGIC
# MAGIC - [Unity Catalog Model Management](https://docs.databricks.com/machine-learning/manage-model-lifecycle/)
# MAGIC - [Model Serving Endpoints](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints)
# MAGIC - [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html)
# MAGIC - [Model Signatures](https://mlflow.org/docs/latest/models.html#model-signature)
# MAGIC - [PyFunc Models](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html)
# MAGIC
# MAGIC ### Best Practices Implemented
# MAGIC
# MAGIC 1. **Configuration Management**: All user-configurable values at the top after package installation
# MAGIC 2. **Dependency Management**: Package installation before configuration to avoid kernel restart issues
# MAGIC 3. **Error Handling**: Comprehensive try-catch blocks for robustness
# MAGIC 4. **Documentation**: Clear markdown explanations for each step
# MAGIC 5. **Testing**: Automated endpoint testing with proper data formats
# MAGIC 6. **Security**: Proper authentication token usage
# MAGIC 7. **Scalability**: GPU-enabled endpoints with auto-scaling
# MAGIC 8. **Monitoring**: Status polling and logging throughout the process
# MAGIC
# MAGIC ### Customization Options
# MAGIC
# MAGIC - **Model Architecture**: Replace CLIP with other embedding models
# MAGIC - **Endpoint Configuration**: Adjust workload sizes and scaling settings
# MAGIC - **Testing Data**: Use your own images/text for validation
# MAGIC - **Monitoring**: Add inference tables and performance tracking
# MAGIC - **Model Routes**: Add more model versions with different names
# MAGIC
# MAGIC ### Troubleshooting Tips
# MAGIC
# MAGIC 1. **Permission Issues**: Ensure proper Unity Catalog privileges
# MAGIC 2. **Endpoint Failures**: Check logs in Databricks Serving UI
# MAGIC 3. **Model Loading**: Verify all dependencies in pip_requirements
# MAGIC 4. **Memory Issues**: Consider larger workload sizes for complex models
# MAGIC 5. **Network Timeouts**: Increase wait times for large model deployments
# MAGIC 6. **Package Conflicts**: Use compatible versions as specified in the notebook
# MAGIC 7. **Empty Embeddings**: Ensure data is sent in correct DataFrame format

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Success Criteria Checklist
# MAGIC
# MAGIC ✅ **Model Development**
# MAGIC - [ ] CLIP image embedding model implemented
# MAGIC - [ ] CLIP image+text embedding model implemented  
# MAGIC - [ ] Models tested locally before registration
# MAGIC - [ ] MLflow 2.11+ compatible signatures used
# MAGIC
# MAGIC ✅ **Unity Catalog Integration**
# MAGIC - [ ] Models registered in Unity Catalog with proper naming
# MAGIC - [ ] Model signatures defined and validated
# MAGIC - [ ] Model versions properly documented
# MAGIC
# MAGIC ✅ **Model Serving**
# MAGIC - [ ] Unified endpoint created with named routes
# MAGIC - [ ] Both models accessible through single endpoint
# MAGIC - [ ] Endpoints return valid embeddings
# MAGIC - [ ] Proper DataFrame format used for requests
# MAGIC
# MAGIC ✅ **Lifecycle Management**
# MAGIC - [ ] Model aliases configured for easy reference
# MAGIC - [ ] Model descriptions updated appropriately
# MAGIC
# MAGIC ✅ **Production Readiness**
# MAGIC - [ ] Error handling implemented
# MAGIC - [ ] Endpoints configured for auto-scaling
# MAGIC - [ ] Authentication properly configured
# MAGIC - [ ] Documentation and cleanup procedures provided
# MAGIC - [ ] Package dependencies properly managed
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🎉 Congratulations! You have successfully deployed CLIP embedding models with Unity Catalog integration and proper MLOps practices.**
