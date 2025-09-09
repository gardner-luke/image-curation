# Image Curation Pipeline

A comprehensive ML pipeline for processing crop images, registering CLIP models, and generating embeddings for vector search and similarity matching.

## Overview

This pipeline enables you to:
- Process large directories of crop images and extract metadata
- Register CLIP models for both image and text embeddings
- Generate embeddings for images using the registered model
- Store everything in Delta tables for efficient querying and vector search

## Pipeline Flow

### 1. Image Directory Processing and Metadata Storage
**File:** `Validated/Crop Image Directory Processing and Metadata Storage.ipynb`

**Purpose:** Scans and processes image directories to extract metadata and prepare images for embedding generation.

**What it does:**
- Recursively scans `/Volumes/autobricks/agriculture/crop_images` for JPEG files
- Extracts file metadata (name, path, size, creation date)
- Converts images to base64 encoding for ML model processing
- Stores all data in a Delta table: `autobricks.agriculture.crop_images_directory`

**Key outputs:**
- File path and name
- Folder structure
- File size and creation timestamp
- Base64-encoded image data

### 2. CLIP Model Registration
**File:** `Validated/Register CLIP Model.ipynb`

**Purpose:** Registers a CLIP model in MLflow/Unity Catalog for both image and text embedding generation.

**What it does:**
- Creates a custom MLflow PyFunc model using OpenAI's CLIP-ViT-Large-Patch14
- Supports both image and text embedding generation
- Handles base64-encoded images and raw text input
- Registers the model in Unity Catalog: `autobricks.agriculture.clip_embedding-356`
- Creates a serving endpoint for real-time inference

**Key features:**
- GPU acceleration when available
- Automatic image format conversion (RGB)
- Flexible input handling (image or text)
- Production-ready serving configuration

### 3. Batch Embedding Generation
**File:** `Validated/Batch CLIP Embedding for Crop Images with Delta Tables.ipynb`

**Purpose:** Processes the stored images through the registered CLIP model to generate embeddings.

**What it does:**
- Loads the registered CLIP model from Unity Catalog
- Creates a Spark UDF for distributed embedding generation
- Processes images from the directory table
- Generates 768-dimensional embeddings for each image
- Saves results to: `autobricks.agriculture.crop_images_directory_embeddings`

**Key outputs:**
- Original image metadata
- 768-dimensional CLIP embeddings
- Ready for vector similarity search

## Usage

### Prerequisites
- Databricks workspace with Unity Catalog access
- Access to `/Volumes/autobricks/agriculture/crop_images`
- MLflow and PyTorch installed

### Running the Pipeline

1. **Process Images:** Run the first notebook to scan and store image metadata
2. **Register Model:** Run the second notebook to register the CLIP model
3. **Generate Embeddings:** Run the third notebook to create embeddings

### Configuration

Update these variables in each notebook as needed:
- `CATALOG_NAME`: Your Unity Catalog name
- `SCHEMA_NAME`: Your schema name  
- `MODEL_NAME`: Your model name
- `ENDPOINT_NAME`: Your serving endpoint name

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