# Databricks notebook source
# MAGIC %md
# MAGIC # CLIP Model Serving with Unity Catalog
# MAGIC
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Create CLIP embedding models (image-only and image+text) using MLflow PyFunc
# MAGIC 2. Register models in Unity Catalog with proper versioning
# MAGIC 3. Create model serving endpoints
# MAGIC 4. Test the deployed models
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

# Install required packages
%pip install transformers torch torchvision pillow mlflow>=2.15.1
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
MODEL_BASE_NAME = "clip_embedding_model"
ENDPOINT_BASE_NAME = "clip-embedding-endpoint"

# Data source configuration (optional - for testing)
SAMPLE_IMAGE_TABLE = "autobricks.agriculture.crop_images_directory"  # Optional: table with base64 encoded images

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Install Dependencies and Setup

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

# COMMAND ----------

# Image-only embedding model
class CLIP_IMAGE_EMBEDDING(mlflow.pyfunc.PythonModel):
    """
    CLIP model for image embeddings only.
    Processes base64-encoded images and returns embedding vectors.
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

    def predict(self, context, df):
        """Make predictions on a DataFrame containing base64 encoded images."""
        return df['model_input'].apply(lambda x: self._get_image_embedding_bytearray(x))

# COMMAND ----------

# Image and text embedding model  
class CLIP_IMAGE_TEXT_EMBEDDING(mlflow.pyfunc.PythonModel):
    """
    CLIP model for both image and text embeddings.
    Can process either images (base64) or text based on input_type parameter.
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

    def predict(self, context, df, params):
        """Make predictions based on input type (image or text)."""
        input_type = params.get('input_type')
        
        if input_type.lower() == 'image':
            print('🖼️ Processing image embeddings')
            return df['model_input'].apply(lambda x: self._get_image_embedding_bytearray(x))
        elif input_type.lower() == 'text':
            print('📝 Processing text embeddings')
            return df['model_input'].apply(lambda x: self._get_text_embedding(x))
        else:
            raise ValueError("input_type must be either 'image' or 'text'")

print("✅ Model classes defined successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Create Sample Data and Model Signatures

# COMMAND ----------

# Create sample data for testing and signature inference
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

# Generate sample data
sample_image_base64 = create_sample_image_data()
sample_text = "a red square image"

# Create test DataFrames
image_test_df = pd.DataFrame({'model_input': [sample_image_base64, sample_image_base64]})
text_test_df = pd.DataFrame({'model_input': [sample_text, "hello world"]})

print("✅ Sample data created successfully!")
print(f"📁 Image data shape: {image_test_df.shape}")
print(f"📝 Text data shape: {text_test_df.shape}")

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
    
    # Test prediction to create signature
    test_result = image_model.predict(context=None, df=image_test_df)
    
    # Infer signature
    signature = infer_signature(image_test_df, test_result)
    
    # Define pip requirements
    pip_requirements = [
        "mlflow>=2.15.1",
        "torch>=2.3.1+cu121",
        "torchvision>=0.18.1+cu121", 
        "transformers>=4.41.2",
        "accelerate>=0.31.0",
        "pillow",
        "pandas>=1.5.3"
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
    
    # Test both image and text predictions
    image_test_result = multimodal_model.predict(context=None, df=image_test_df, params={'input_type': 'image'})
    text_test_result = multimodal_model.predict(context=None, df=text_test_df, params={'input_type': 'text'})
    
    # Infer signature (using image data as primary example)
    signature = infer_signature(image_test_df, image_test_result, params={'input_type': 'image'})
    
    # Define pip requirements
    pip_requirements = [
        "mlflow>=2.15.1", 
        "torch>=2.3.1+cu121",
        "torchvision>=0.18.1+cu121",
        "transformers>=4.41.2",
        "accelerate>=0.31.0",
        "pillow",
        "pandas>=1.5.3"
    ]
    
    # Log the model
    model_info = mlflow.pyfunc.log_model(
        artifact_path="clip_image_text",
        python_model=multimodal_model,
        registered_model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}",
        signature=signature,
        pip_requirements=pip_requirements,
        input_example=image_test_df,
        metadata={
            "model_type": "multimodal_embedding",
            "base_model": "openai/clip-vit-large-patch14", 
            "description": "CLIP model for both image and text embeddings",
            "version": "2.0",
            "capabilities": ["image_embedding", "text_embedding"],
            "embedding_dimension": 768,
            "input_types": ["image", "text"]
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
# MAGIC - **Champion**: The current production model (Version 1 initially)
# MAGIC - **Challenger**: The model being tested for promotion (Version 2)

# COMMAND ----------

# Initialize MLflow client for alias management
client = mlflow.MlflowClient()
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}"

# Set Version 1 as Champion (current production model)
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version="1"
)

# Set Version 2 as Challenger (testing for promotion)
client.set_registered_model_alias(
    name=model_name,
    alias="challenger", 
    version="2"
)

# Update model descriptions
client.update_model_version(
    name=model_name,
    version="1",
    description="Image-only CLIP embedding model - Production Champion"
)

client.update_model_version(
    name=model_name,
    version="2", 
    description="Enhanced CLIP model with image and text capabilities - Testing Challenger"
)

print(f"✅ Model aliases configured successfully!")
print(f"🏆 Champion: {model_name} version 1 (image-only)")
print(f"🥇 Challenger: {model_name} version 2 (image + text)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Create Model Serving Endpoints
# MAGIC
# MAGIC We'll create serving endpoints for both versions of our model.

# COMMAND ----------

# Initialize deployment client
deploy_client = get_deploy_client("databricks")

print("🔧 Creating model serving endpoints...")

# COMMAND ----------

# Create endpoint for Champion model (Version 1 - Image only)
champion_endpoint_name = f"{ENDPOINT_BASE_NAME}-champion"

try:
    champion_endpoint = deploy_client.create_endpoint(
        name=champion_endpoint_name,
        config={
            "served_entities": [
                {
                    "entity_name": f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}",  # model_name-version
                    "entity_version": "1",
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                    "workload_type": "GPU_SMALL"
                }
            ],
            "traffic_config": {
                "routes": [
                    {
                        "served_model_name": f"{MODEL_BASE_NAME}-1",  # must match entity_name above
                        "traffic_percentage": 100
                    }
                ]
            }
        }
    )
    
    print(f"✅ Champion endpoint created: {champion_endpoint_name}")
    
except Exception as e:
    if "already exists" in str(e):
        print(f"ℹ️ Champion endpoint {champion_endpoint_name} already exists")
    else:
        print(f"❌ Error creating champion endpoint: {str(e)}")

# COMMAND ----------

# Create endpoint for Challenger model (Version 2 - Image + Text)
challenger_endpoint_name = f"{ENDPOINT_BASE_NAME}-challenger"

try:
    challenger_endpoint = deploy_client.create_endpoint(
        name=challenger_endpoint_name,
        config={
            "served_entities": [
                {
                    "entity_name": f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_BASE_NAME}",  # model_name-version
                    "entity_version": "2",
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True,
                    "workload_type": "GPU_SMALL"
                }
            ],
            "traffic_config": {
                "routes": [
                    {
                        "served_model_name": f"{MODEL_BASE_NAME}-2",  # must match entity_name-version above
                        "traffic_percentage": 100
                    }
                ]
            }
        }
    )
    
    print(f"✅ Challenger endpoint created: {challenger_endpoint_name}")
    
except Exception as e:
    if "already exists" in str(e):
        print(f"ℹ️ Challenger endpoint {challenger_endpoint_name} already exists")
    else:
        print(f"❌ Error creating challenger endpoint: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⏳ Wait for Endpoints to be Ready

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

# Wait for both endpoints
print("🚀 Waiting for endpoints to be ready...")
champion_ready = wait_for_endpoint_ready(champion_endpoint_name)
challenger_ready = wait_for_endpoint_ready(challenger_endpoint_name)

if champion_ready and challenger_ready:
    print("🎉 All endpoints are ready for testing!")
else:
    print("⚠️ Some endpoints may not be ready. Check the Databricks UI for details.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧪 Test Model Endpoints
# MAGIC
# MAGIC Let's test both endpoints to ensure they're working correctly.

# COMMAND ----------

# Get authentication token
notebook_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Test Champion endpoint (Image-only)
def test_champion_endpoint():
    """Test the champion endpoint with image data."""
    print("🧪 Testing Champion endpoint (Image-only model)...")
    
    endpoint_url = f"{WORKSPACE_URL}/serving-endpoints/{champion_endpoint_name}/invocations"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {notebook_token}"
    }
    
    # Prepare test data
    test_data = {
        "inputs": [sample_image_base64]
    }
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Champion endpoint test successful!")
            print(f"📊 Response shape: {np.array(result['predictions'][0]).shape}")
            return True
        else:
            print(f"❌ Champion endpoint test failed: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing champion endpoint: {str(e)}")
        return False

# Test Challenger endpoint (Image + Text)
def test_challenger_endpoint():
    """Test the challenger endpoint with both image and text data."""
    print("🧪 Testing Challenger endpoint (Image + Text model)...")
    
    endpoint_url = f"{WORKSPACE_URL}/serving-endpoints/{challenger_endpoint_name}/invocations"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {notebook_token}"
    }
    
    # Test with image data
    print("🖼️ Testing with image input...")
    image_test_data = {
        "inputs": [sample_image_base64],
        "params": {"input_type": "image"}
    }
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=image_test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Challenger image test successful!")
            print(f"📊 Image embedding shape: {np.array(result['predictions'][0]).shape}")
        else:
            print(f"❌ Challenger image test failed: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing challenger endpoint with image: {str(e)}")
        return False
    
    # Test with text data
    print("📝 Testing with text input...")
    text_test_data = {
        "inputs": ["a beautiful sunset over the mountains"],
        "params": {"input_type": "text"}
    }
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=text_test_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Challenger text test successful!")
            print(f"📊 Text embedding shape: {np.array(result['predictions'][0]).shape}")
            return True
        else:
            print(f"❌ Challenger text test failed: {response.status_code}")
            print(f"📋 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing challenger endpoint with text: {str(e)}")
        return False

# Run tests
if champion_ready:
    champion_test_result = test_champion_endpoint()
else:
    print("⚠️ Skipping champion test - endpoint not ready")
    champion_test_result = False

if challenger_ready:
    challenger_test_result = test_challenger_endpoint()
else:
    print("⚠️ Skipping challenger test - endpoint not ready")
    challenger_test_result = False

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

print(f"\n🚀 Serving Endpoints:")
print(f"   • Champion: {champion_endpoint_name}")
print(f"   • Challenger: {challenger_endpoint_name}")

print(f"\n🔗 Model Versions:")
print(f"   • Version 1: Image-only embeddings")
print(f"   • Version 2: Image + Text embeddings")

print(f"\n🏷️ Current Aliases:")
try:
    aliases = client.get_model_version_by_alias(name=model_name, alias="champion")
    print(f"   • Champion: Version {aliases.version}")
except:
    print(f"   • Champion: Version 1")

try:
    aliases = client.get_model_version_by_alias(name=model_name, alias="challenger")
    print(f"   • Challenger: Version {aliases.version}")
except:
    print(f"   • Challenger: Version 2")

print(f"\n📋 Next Steps:")
print(f"   1. Monitor endpoint performance in Databricks UI")
print(f"   2. Set up inference tables for monitoring (optional)")
print(f"   3. Configure endpoint permissions as needed")
print(f"   4. Test endpoints with your actual data")
print(f"   5. Set up automated retraining pipelines")

print(f"\n🔍 Useful URLs:")
print(f"   • Serving UI: {WORKSPACE_URL}/#mlflow/serving")
print(f"   • Models: {WORKSPACE_URL}/#mlflow/models")
print(f"   • Experiments: {WORKSPACE_URL}/#mlflow/experiments")

print("\n" + "="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧹 Optional: Cleanup Resources
# MAGIC
# MAGIC Uncomment and run the following cell to clean up endpoints and models when no longer needed.

# COMMAND ----------

# # UNCOMMENT TO ENABLE CLEANUP - BE CAREFUL!
# # This will delete your endpoints and models

def cleanup_resources():
    """Clean up serving endpoints and models."""
    print("🧹 Starting cleanup process...")
    
    # Delete serving endpoints
    try:
        deploy_client.delete_endpoint(endpoint=champion_endpoint_name)
        print(f"🗑️ Deleted champion endpoint: {champion_endpoint_name}")
    except Exception as e:
        print(f"⚠️ Could not delete champion endpoint: {str(e)}")
    
    try:
        deploy_client.delete_endpoint(endpoint=challenger_endpoint_name)
        print(f"🗑️ Deleted challenger endpoint: {challenger_endpoint_name}")
    except Exception as e:
        print(f"⚠️ Could not delete challenger endpoint: {str(e)}")
    
    #Delete model versions (optional - be very careful!)
    try:
        client.delete_model_version(name=model_name, version="1")
        client.delete_model_version(name=model_name, version="2")
        client.delete_registered_model(name=model_name)
        print(f"🗑️ Deleted model: {model_name}")
    except Exception as e:
        print(f"⚠️ Could not delete model: {str(e)}")
    
    print("✅ Cleanup completed!")

#Uncomment the line below to run cleanup
cleanup_resources()

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
# MAGIC    - Model signatures for input/output validation
# MAGIC    - Deployment using MLflow Deployments SDK
# MAGIC
# MAGIC 3. **Model Lifecycle Management**
# MAGIC    - Champion/Challenger pattern for testing
# MAGIC    - Model aliases for version management
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
# MAGIC 1. **Configuration Management**: All user-configurable values at the top
# MAGIC 2. **Error Handling**: Comprehensive try-catch blocks for robustness
# MAGIC 3. **Documentation**: Clear markdown explanations for each step
# MAGIC 4. **Testing**: Automated endpoint testing
# MAGIC 5. **Security**: Proper authentication token usage
# MAGIC 6. **Scalability**: GPU-enabled endpoints with auto-scaling
# MAGIC 7. **Monitoring**: Status polling and logging throughout the process
# MAGIC
# MAGIC ### Customization Options
# MAGIC
# MAGIC - **Model Architecture**: Replace CLIP with other embedding models
# MAGIC - **Endpoint Configuration**: Adjust workload sizes and scaling settings
# MAGIC - **Testing Data**: Use your own images/text for validation
# MAGIC - **Monitoring**: Add inference tables and performance tracking
# MAGIC
# MAGIC ### Troubleshooting Tips
# MAGIC
# MAGIC 1. **Permission Issues**: Ensure proper Unity Catalog privileges
# MAGIC 2. **Endpoint Failures**: Check logs in Databricks Serving UI
# MAGIC 3. **Model Loading**: Verify all dependencies in pip_requirements
# MAGIC 4. **Memory Issues**: Consider larger workload sizes for complex models
# MAGIC 5. **Network Timeouts**: Increase wait times for large model deployments

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
# MAGIC
# MAGIC ✅ **Unity Catalog Integration**
# MAGIC - [ ] Models registered in Unity Catalog with proper naming
# MAGIC - [ ] Model signatures defined and validated
# MAGIC - [ ] Model versions properly documented
# MAGIC
# MAGIC ✅ **Model Serving**
# MAGIC - [ ] Champion endpoint created and tested
# MAGIC - [ ] Challenger endpoint created and tested
# MAGIC - [ ] Both endpoints return valid embeddings
# MAGIC
# MAGIC ✅ **Lifecycle Management**
# MAGIC - [ ] Model aliases configured (champion/challenger)
# MAGIC - [ ] Model descriptions updated appropriately
# MAGIC
# MAGIC ✅ **Production Readiness**
# MAGIC - [ ] Error handling implemented
# MAGIC - [ ] Endpoints configured for auto-scaling
# MAGIC - [ ] Authentication properly configured
# MAGIC - [ ] Documentation and cleanup procedures provided
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **🎉 Congratulations! You have successfully deployed CLIP embedding models with Unity Catalog integration and proper MLOps practices.**
