import streamlit as st
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks import sql
from databricks.sdk.core import Config
import os

st.set_page_config(layout="wide", page_title="Vector Search App")

# Initialize Databricks client
w = WorkspaceClient()

def execute_sql_query(query: str) -> pd.DataFrame:
    """Execute SQL query using Databricks SQL connector"""
    cfg = Config()
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID', '8baced1ff014912d')}",
        credentials_provider=lambda: cfg.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

def search_images(query: str, selected_filters: list):
    """Complete search flow: generate embedding and search vector database"""
    try:
        # Step 1: Generate embedding
        with st.spinner("Generating text embedding..."):
            sql_query = f"""
            SELECT 
                '{query}' AS input_text,
                ai_query('clip-text-embedding', request => '{query}') AS embedding
            """
            result_df = execute_sql_query(sql_query)
            
            if result_df.empty or 'embedding' not in result_df.columns:
                st.error("Failed to generate embedding")
                return None
                
            embedding = result_df['embedding'].iloc[0]
        
        # Step 2: Search vector database
        with st.spinner("Searching for similar images..."):
            # For now, search all results and filter client-side
            # TODO: Implement server-side filtering when query_index supports filters
            query_result = w.vector_search_indexes.query_index(
                index_name="autobricks.agriculture.crop_images_directory_embeddings_index",
                columns=["file_path", "file_name", "folder"],
                query_vector=embedding,
                num_results=50,  # Get more results to filter from
            )
            
            # Apply client-side filtering based on selected crop types
            if selected_filters and query_result and query_result.result.data_array:
                filtered_results = []
                for result in query_result.result.data_array:
                    # Extract folder information from result
                    folder_value = None
                    if isinstance(result, dict) and 'folder' in result:
                        folder_value = str(result['folder']).lower()
                    elif isinstance(result, (list, tuple)) and len(result) > 2:
                        # Assuming folder is the third column (index 2)
                        folder_value = str(result[2]).lower()
                    
                    # Check if folder matches any selected filter
                    if folder_value and any(crop.lower() in folder_value for crop in selected_filters):
                        filtered_results.append(result)
                
                # Update the result with filtered data
                query_result.result.data_array = filtered_results[:12]  # Limit to 12 results
            
            return query_result
            
    except Exception as e:
        st.error(f"Search failed: {e}")
        return None

def display_search_results(query_result, search_query, selected_filters):
    """Display search results including images"""
    if not query_result or not query_result.result.data_array:
        st.warning("No similar images found")
        return
    
    # Process results data
    results_data = []
    for i, result in enumerate(query_result.result.data_array):
        result_dict = {'Rank': i + 1}
        
        # Handle different result formats
        if isinstance(result, dict):
            for key in ['file_path', 'file_name', 'folder']:
                if key in result:
                    result_dict[key] = str(result[key])
        elif isinstance(result, (list, tuple)):
            columns = ["file_path", "file_name", "folder"]
            for j, value in enumerate(result):
                if j < len(columns):
                    col_name = columns[j]
                    result_dict[col_name] = str(value)
        else:
            result_dict['Data'] = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            
        results_data.append(result_dict)
    
    results_df = pd.DataFrame(results_data)
    
    # Display success message with applied filters
    if selected_filters:
        filters_text = ", ".join(selected_filters)
        st.success(f"✅ Found {len(query_result.result.data_array)} similar images for: '{search_query}' (filtered by: {filters_text})")
    else:
        st.success(f"✅ Found {len(query_result.result.data_array)} similar images for: '{search_query}'")
    
    # Display images first
    if not results_df.empty and 'file_path' in results_df.columns:
        st.subheader("📸 Similar Images")
        
        # Filter for image files
        image_paths = results_df['file_path'].tolist()
        image_files = [path for path in image_paths if path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        if image_files:
            st.write(f"**Displaying {len(image_files)} images:**")
            
            # Create columns for grid layout (3 images per row)
            cols_per_row = 3
            for i in range(0, len(image_files), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j, col in enumerate(cols):
                    if i + j < len(image_files):
                        image_path = image_files[i + j]
                        
                        with col:
                            try:
                                # Download image from Databricks volume
                                response = w.files.download(image_path)
                                image_data = response.contents.read()
                                
                                # Display image
                                st.image(image_data, caption=os.path.basename(image_path), use_column_width=True)
                                
                            except Exception as img_error:
                                st.error(f"Error loading image: {os.path.basename(image_path)}")
        else:
            st.info("No image files found in the results")
    
    # Display results table below images
    st.subheader("📋 Search Results")
    st.dataframe(results_df, use_container_width=True)

# Sidebar
with st.sidebar:
    st.header("🔍 Search & Filter")
    
    # Search interface
    search_query = st.text_input(
        label="What are you looking for?",
        placeholder="e.g., red objects, outdoor scenes, vehicles...",
        key="search_input"
    )
    
    # Metadata filter with checkboxes
    st.subheader("🏷️ Filter by Metadata")
    filter_options = ["Maize", "Sugarcane", "Wheat", "Rice", "Jute"]
    selected_filters = []
    
    for filter_option in filter_options:
        if st.checkbox(filter_option, key=f"filter_{filter_option.lower().replace(' ', '_')}"):
            selected_filters.append(filter_option)
    
    # Show selected filters
    if selected_filters:
        st.write(f"**Selected:** {', '.join(selected_filters)}")
    else:
        st.write("**Selected:** All categories")
    
    # Search button
    search_clicked = st.button("🔍 Search Images", type="primary", use_container_width=True)
    
    # Status indicator
    st.markdown("---")
    st.subheader("📊 Status")
    
    if search_clicked:
        if search_query:
            st.info("🔍 Scanning for similarity...")
            # Execute search and store results
            query_result = search_images(search_query, selected_filters)
            st.session_state.current_query = search_query
            st.session_state.current_filters = selected_filters
            st.session_state.current_results = query_result
        else:
            st.warning("Please enter a search query")
    
    # Show current search status
    if 'current_results' in st.session_state and st.session_state.current_results is not None:
        st.success("✅ Search completed")
        st.write(f"**Query:** {st.session_state.current_query}")
        if st.session_state.current_filters:
            st.write(f"**Filter:** {', '.join(st.session_state.current_filters)}")
        else:
            st.write("**Filter:** All categories")
    elif search_clicked and search_query:
        st.info("🔍 Scanning for similarity...")
    else:
        st.info("Ready to search")

# Main content area
st.header("Databricks Image Curation")
st.markdown("Find similar images from your database using natural language descriptions and metadata filters.")

# Add helpful information section (always visible)
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🔍 How it works:**
    1. Enter your search description
    2. Select metadata filters (optional)
    3. Click search to find similar images
    4. Browse results and images
    """)

with col2:
    st.markdown("""
    **💡 Search Tips:**
    - Be descriptive: "red sports car"
    - Use specific terms: "outdoor landscape"
    - Try different angles: "aerial view of buildings"
    - Use color and object combinations
    """)

with col3:
    st.markdown("""
    **🔄 Image Deduplication:**
    - Identify duplicate images automatically
    - Find images with high similarity scores
    - Detect near-identical content
    - Clean up your dataset efficiently
    - Reduce storage and processing costs
    """)

# Display search results if available
if 'current_results' in st.session_state and st.session_state.current_results is not None:
    display_search_results(
        st.session_state.current_results, 
        st.session_state.current_query, 
        st.session_state.current_filters
    ) 