# Databricks Image Search App - Implementation Specification

## Application Overview
Build a FastAPI-based web application that enables users to search for similar images using text descriptions. The app converts text queries to embeddings via a CLIP model, performs vector similarity search, and displays matching images from Unity Catalog volumes in a responsive grid. This follows the same architectural pattern as your existing todo app but focuses on image search functionality.

## Architecture Flow

User Text Input (Web UI) → FastAPI Backend → CLIP Model Serving Endpoint → Text Embedding → Vector Search Index → Similar Image Embeddings → Unity Catalog Volume File Paths → JPEG Image Display Grid


## Core Components

### 1. Main FastAPI Application (`app.py`)
**Purpose**: Primary user interface and orchestration

**Key Functionality**:
- Text input field for search queries
- Integration with embedding service to convert text to vectors
- Vector similarity search execution
- Image grid display with responsive layout
- Loading states and error handling
- Basic pagination for large result sets

**UI Elements**:
- Search input box with submit functionality
- Results counter and search metadata
- Grid layout displaying images with thumbnails
- Basic image metadata overlay (filename, similarity score)
- Simple navigation controls

### 2. Embedding Service Client (`src/embedding_client.py`)
**Purpose**: Interface with the CLIP model serving endpoint

**Core Responsibilities**:
- Make HTTP requests to the model serving endpoint
- Handle authentication using Databricks SDK credentials
- Convert text strings to embedding vectors
- Implement basic retry logic for API failures
- Cache embeddings for repeated queries

**Key Methods**:
- Initialize connection to model serving endpoint
- Submit text for embedding generation
- Handle API responses and error states
- Return embedding vectors in the expected format

### 3. Vector Search Client (`src/vector_search_client.py`)
**Purpose**: Interface with Databricks Vector Search index

**Core Responsibilities**:
- Connect to the specified vector search index
- Execute similarity search queries using embedding vectors
- Process search results and extract metadata
- Handle search errors and empty results
- Return ranked results with similarity scores

**Key Methods**:
- Initialize vector search client connection
- Execute similarity search with configurable result limits
- Parse search results and extract file paths
- Handle search index connectivity issues

### 4. Image Handler (`src/image_utils.py`)
**Purpose**: Load and process images from Unity Catalog volumes

**Core Responsibilities**:
- Access images stored in Unity Catalog volumes using file paths
- Load JPEG images and convert to displayable format
- Generate thumbnails for grid display
- Handle missing or corrupted image files gracefully
- Optimize image loading for performance

**Key Methods**:
- Load individual images from volume paths
- Batch load multiple images efficiently
- Create consistent thumbnail sizes
- Validate image file existence and format

### 5. Configuration Management (`src/config.py`)
**Purpose**: Centralize application configuration

**Configuration Items**:
- Model serving endpoint name/URL
- Vector search index identifier
- Unity Catalog volume path for images
- Display settings (grid columns, image sizes)
- Performance settings (cache timeouts, batch sizes)
- Default search parameters

## Technical Requirements

### Dependencies
- `databricks-sdk` - Databricks platform integration
- `fastapi` - Web application framework
- `pillow` - Image processing
- `pandas` - Data manipulation
- `requests` - HTTP client operations

### Authentication & Permissions
Use Databricks Apps built-in authentication

Service principal requires:
- Execute permissions on model serving endpoint
- Query permissions on vector search index
- Read permissions on Unity Catalog volume
- Basic Unity Catalog access (catalog, schema usage)

### File Structure (adapted for your existing repo)

app.py                     # Main FastAPI application (extend existing)
requirements.txt          # Add new dependencies to existing file
.env                       # Add image search environment variables
templates/
  ├── image_search.html    # Main image search interface
  └── base.html           # Shared template base
static/
  ├── css/
  │   └── image_search.css # Styling for image search UI
  ├── js/
  │   └── image_search.js  # Frontend JavaScript logic
  └── images/             # Static image assets
src/
  ├── embedding_client.py  # Model serving integration
  ├── vector_search_client.py # Vector search integration
  ├── image_utils.py      # Image loading utilities
  └── config.py           # Configuration management (extend existing)


## Implementation Approach

### Core Application Logic
1. Initialize Databricks SDK clients and connections
2. Create FastAPI interface with text input field
3. On search submission:
   - Call embedding client to convert text to vector
   - Submit vector to search client for similarity search
   - Extract file paths from search results
   - Load images from Unity Catalog volumes using file paths
   - Display images in responsive grid layout

### Error Handling Strategy
- Graceful handling of model serving failures
- Fallback for vector search connectivity issues
- Missing image file management
- User-friendly error messages
- Automatic retry mechanisms where appropriate

### Performance Considerations
Implement caching for:
- Database connections
- Generated embeddings
- Loaded images

Additional optimizations:
- Lazy loading for large image sets
- Thumbnail generation for faster display
- Connection pooling for external services

### User Experience
- Real-time loading indicators during search
- Progressive image loading in grid
- Responsive design for different screen sizes
- Clear feedback for empty results or errors
- Simple, intuitive interface design

## Integration Points

### Model Serving Endpoint
- Standard HTTP API calls using Databricks SDK
- JSON payload with text input
- Response parsing for embedding vectors
- Authentication via SDK credential provider

### Vector Search Index
- Databricks Vector Search client integration
- Similarity search with embedding vectors
- Configurable result limits and filtering
- Metadata extraction from search results

### Unity Catalog Volumes
- File system access using volume paths (`/Volumes/catalog/schema/volume/path`)
- Standard file I/O operations for image loading
- Path validation and error handling
- Support for nested directory structures

## Data Flow

1. **Input Processing**: User enters text query in UI
2. **Embedding Generation**: Text sent to CLIP model serving endpoint, returns embedding vector
3. **Similarity Search**: Embedding vector queried against vector search index, returns similar embeddings with metadata
4. **Path Extraction**: File paths extracted from search result metadata
5. **Image Loading**: Images loaded from Unity Catalog volumes using extracted paths
6. **Display**: Images rendered in responsive grid with metadata overlay

## Environment Variables

- `DATABRICKS_HOST` - Workspace URL
- `MODEL_SERVING_ENDPOINT` - Name of CLIP model endpoint
- `VECTOR_SEARCH_INDEX` - Full name of vector search index
- `IMAGE_VOLUME_PATH` - Base path to Unity Catalog volume containing images

## Key Success Criteria

- Successful text-to-embedding conversion via model serving
- Functional vector similarity search with relevant results
- Reliable image loading from Unity Catalog volumes
- Responsive grid display of search results
- Proper error handling and user feedback
- Performance suitable for interactive use