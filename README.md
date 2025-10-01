# Image Curation Pipeline

A comprehensive ML pipeline for processing image datasets, registering CLIP models, and generating embeddings for vector search, similarity matching, and dataset deduplication.

<img width="1645" height="775" alt="Screenshot 2025-09-30 at 10 30 15 PM" src="https://github.com/user-attachments/assets/5b1c325a-3ec6-478b-849f-23be8e687429" />

## Overview

This pipeline enables you to:
- Process large directories of images and extract metadata
- Register CLIP models for both image and text embeddings
- Generate embeddings for images using the registered model
- Store everything in Delta tables for efficient querying and vector search
- **Curate high-quality datasets** through interactive search and deduplication
- **Identify duplicate images** automatically using similarity scores

## Pipeline Flow

### 1. Image Directory Processing and Metadata Storage
**File:** `production/00 - Crop Image Directory Processing and Metadata Storage.ipynb`

**Purpose:** Scans and processes image directories to extract metadata and prepare images for embedding generation.

**What it does:**
- Recursively scans your image directory for supported image files (JPEG, PNG, etc.)
- Extracts file metadata (name, path, size, creation date)
- Converts images to base64 encoding for ML model processing
- Stores all data in a Delta table for further processing

**Key outputs:**
- File path and name
- Folder structure (useful for categorization)
- File size and creation timestamp
- Base64-encoded image data

**Example:** The provided notebook processes `/Volumes/autobricks/agriculture/crop_images` as an example, but you can point it to any image directory.

### 2. CLIP Model Registration
**Files:** 
- `production/01a - Register CLIP Text Embedding Model.ipynb`
- `production/01b - Register CLIP Image Embedding Model.ipynb`

**Purpose:** Registers CLIP models in MLflow/Unity Catalog for both image and text embedding generation.

**What it does:**
- Creates custom MLflow PyFunc models using OpenAI's CLIP-ViT-Large-Patch14
- Supports both image and text embedding generation
- Handles base64-encoded images and raw text input
- Registers the models in Unity Catalog
- Creates serving endpoints for real-time inference

**Key features:**
- GPU acceleration when available
- Automatic image format conversion (RGB)
- Flexible input handling (image or text)
- Production-ready serving configuration

### 3. Batch Embedding Generation
**Files:**
- `production/02a - Batch Inference with Spark UDF.ipynb`
- `production/02b - Test Endpoints with SQL Functions.ipynb`

**Purpose:** Processes the stored images through the registered CLIP model to generate embeddings.

**What it does:**
- Loads the registered CLIP model from Unity Catalog
- Creates a Spark UDF for distributed embedding generation
- Processes images from the directory table
- Generates 768-dimensional embeddings for each image
- Saves results to a dedicated embeddings table

**Key outputs:**
- Original image metadata
- 768-dimensional CLIP embeddings
- Ready for vector similarity search and deduplication

### 4. Vector Search Index Creation
**File:** `production/03 - Create Vector Search Index.ipynb`

**Purpose:** Creates a vector search index for fast similarity search and retrieval.

**What it does:**
- Creates a Databricks Vector Search index
- Enables fast similarity search across your image dataset
- Supports metadata filtering for refined searches
- Optimizes for both search performance and deduplication tasks

## Interactive Search Application

The `interactive_search_app/` directory contains a Streamlit application that provides:

- **Natural language search** of your image dataset
- **Metadata filtering** to narrow down results
- **Visual image grid** display of search results
- **Deduplication capabilities** by identifying high-similarity images
- **Real-time status updates** during search operations

## Usage

### Prerequisites
- Databricks workspace with Unity Catalog access
- Access to your image dataset (local files, cloud storage, or Databricks volumes)
- MLflow and PyTorch installed
- Python environment with required dependencies

### Running the Pipeline

1. **Process Images:** Run the first notebook to scan and store your image metadata
2. **Register Models:** Run the model registration notebooks to set up CLIP models
3. **Generate Embeddings:** Run the batch inference notebooks to create embeddings
4. **Create Vector Index:** Run the vector search index notebook
5. **Launch Search App:** Deploy and run the interactive search application

### Configuration

Update these variables in each notebook for your environment:

**Common Variables:**
- `CATALOG_NAME`: Your Unity Catalog name
- `SCHEMA_NAME`: Your schema name  
- `MODEL_NAME`: Your model name
- `ENDPOINT_NAME`: Your serving endpoint name
- `IMAGE_DIRECTORY`: Path to your image dataset
- `INDEX_NAME`: Your vector search index name

**Example Configuration:**
```python
# Example from agriculture use case
CATALOG_NAME = "autobricks"
SCHEMA_NAME = "agriculture" 
MODEL_NAME = "clip_embedding"
IMAGE_DIRECTORY = "/Volumes/autobricks/agriculture/crop_images"
INDEX_NAME = "crop_images_directory_embeddings_index"
```

## Data Flow

```
Image Directory → Metadata Extraction → Base64 Encoding → Delta Table
                                                              ↓
CLIP Model Registration → Unity Catalog → Serving Endpoint
                                                              ↓
Delta Table → UDF Processing → Embedding Generation → Results Table
```

## Output Tables

- **`crop_images_directory`**: Raw image metadata and base64 data
- **`crop_images_directory_embeddings`**: Images with generated embeddings

## Next Steps

The generated embeddings can be used for:
- Vector similarity search
- Image clustering and classification
- Content-based image retrieval
- Multi-modal search (text-to-image, image-to-image)

## Model Details

- **Model:** OpenAI CLIP-ViT-Large-Patch14
- **Embedding Dimension:** 768
- **Supported Inputs:** Images (JPEG/PNG) and text
- **Framework:** PyTorch with Transformers
