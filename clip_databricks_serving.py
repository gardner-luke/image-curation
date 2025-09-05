# Simple CLIP Model for Databricks Model Serving

import mlflow
import mlflow.pyfunc
import pandas as pd
import base64
from PIL import Image
from io import BytesIO
import torch

class CLIPImageTextEmbedding(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        from transformers import CLIPProcessor, CLIPModel
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def _get_image_embedding(self, base64_image_str):
        # Remove data URL prefix if present
        if base64_image_str.startswith('data:image'):
            base64_image_str = base64_image_str.split(',')[1]
        
        decoded_bytes = base64.b64decode(base64_image_str)
        image = Image.open(BytesIO(decoded_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features.cpu().numpy().tolist()[0]
    
    def _get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        return text_features.cpu().numpy().tolist()[0]
    
    def predict(self, context, model_input, params=None):
        input_type = params.get('input_type')
        input_data = model_input['input_data'][0]  # Assuming single string input
        
        if input_type.lower() == 'image':
            return [self._get_image_embedding(input_data)]
        elif input_type.lower() == 'text':
            return [self._get_text_embedding(input_data)]


def register_clip_model(catalog_name="main", schema_name="models", model_name="clip_embedding"):
    mlflow.set_experiment(f"/Shared/clip_embedding_experiment")
    
    with mlflow.start_run() as run:
        requirements = [
            "torch>=1.9.0",
            "transformers>=4.21.0",
            "Pillow>=8.3.0"
        ]
        
        model = CLIPImageTextEmbedding()
        
        mlflow.pyfunc.log_model(
            artifact_path="clip_model",
            python_model=model,
            pip_requirements=requirements,
            registered_model_name=f"{catalog_name}.{schema_name}.{model_name}"
        )
        
        return f"runs:/{run.info.run_id}/clip_model"


def create_serving_endpoint(model_name, endpoint_name="clip-embedding-endpoint"):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
    
    w = WorkspaceClient()
    
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=model_name,
                    entity_version="1",
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ]
        )
    )


# Main execution
if __name__ == "__main__":
    CATALOG_NAME = "main"
    SCHEMA_NAME = "models" 
    MODEL_NAME = "clip_embedding"
    ENDPOINT_NAME = "clip-embedding-endpoint"
    
    # Register model
    model_uri = register_clip_model(CATALOG_NAME, SCHEMA_NAME, MODEL_NAME)
    
    # Create serving endpoint
    full_model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
    create_serving_endpoint(full_model_name, ENDPOINT_NAME)
    
    print(f"Model registered as: {full_model_name}")
    print(f"Serving endpoint: {ENDPOINT_NAME}")


# Example usage:
"""
# Text embedding
payload = {
    "inputs": {"input_data": ["Hello world"]},
    "params": {"input_type": "text"}
}

# Image embedding  
payload = {
    "inputs": {"input_data": ["data:image/jpeg;base64,/9j/4AAQ..."]},
    "params": {"input_type": "image"}
}
"""