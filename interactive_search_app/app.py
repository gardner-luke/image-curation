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

st.header("Vector Search App 🔍")
st.markdown("Enter text to get raw response from clip-text-embedding model")

# Initialize session state
if 'embedding_result' not in st.session_state:
    st.session_state.embedding_result = None
if 'search_query_text' not in st.session_state:
    st.session_state.search_query_text = ""

# Simple search interface
search_query = st.text_input(
    label="Enter your search query:",
    placeholder="What is Databricks?",
    key="search_query"
)

if st.button("Search", type="primary"):
    if search_query:
        with st.spinner("Calling model via SQL ai_query..."):
            try:
                # Use the same ai_query approach that works in SQL
                sql_query = f"""
                SELECT 
                    '{search_query}' AS input_text,
                    ai_query('clip-text-embedding', request => '{search_query}') AS embedding
                """
                
                st.write("**Executing SQL query:**")
                st.code(sql_query, language="sql")
                
                result_df = execute_sql_query(sql_query)
                
                st.success("✅ SQL query executed successfully!")
                st.subheader("Results:")
                st.dataframe(result_df)
                
                # Show the embedding details and store in session state
                if not result_df.empty and 'embedding' in result_df.columns:
                    embedding = result_df['embedding'].iloc[0]
                    
                    # Store in session state
                    st.session_state.embedding_result = {
                        'embedding': embedding,
                        'search_query': search_query,
                        'result_df': result_df
                    }
                
            except Exception as e:
                st.error(f"❌ Error executing SQL query: {e}")
                st.write("**Note:** Make sure DATABRICKS_WAREHOUSE_ID is set in your environment")
    else:
        st.warning("Please enter a search query")

# Display embedding details if available
if st.session_state.embedding_result:
    embedding_data = st.session_state.embedding_result
    embedding = embedding_data['embedding']
    original_query = embedding_data['search_query']
    
    st.subheader("Embedding Details:")
    st.write(f"**Input Text:** {original_query}")
    st.write(f"**Embedding Type:** {type(embedding)}")
    
    # Try to show embedding info
    if hasattr(embedding, '__len__'):
        st.write(f"**Embedding Length:** {len(embedding)}")
        
    with st.expander("View Raw Embedding"):
        # Convert numpy array to list for JSON display
        if hasattr(embedding, 'tolist'):
            st.json(embedding.tolist())
        else:
            st.json(embedding)
    
    # Add vector search functionality
    st.markdown("---")
    st.subheader("🔍 Vector Search Results")
    
    if st.button("Search Vector Database", type="secondary"):
        with st.spinner("Searching vector database..."):
            try:
                # Query the vector search index with only the columns we want to display
                query_result = w.vector_search_indexes.query_index(
                    index_name="autobricks.agriculture.crop_images_directory_embeddings_index",
                    columns=["file_path", "file_name", "folder"],
                    query_vector=embedding,
                    num_results=10,
                )
                
                st.success("✅ Vector search completed!")
                st.write(f"**Found {len(query_result.result.data_array)} results**")
                
                # Display results
                if query_result.result.data_array:
                    st.subheader("Top 10 Similar Results:")
                    
                    # Show raw results first to debug the structure (limited to prevent browser freeze)
                    with st.expander("🔍 View Raw Vector Search Results (Debug)"):
                        st.write("**Result type:**", type(query_result.result.data_array))
                        st.write("**Number of results:**", len(query_result.result.data_array))
                        if len(query_result.result.data_array) > 0:
                            st.write("**First result type:**", type(query_result.result.data_array[0]))
                            # Limit the display to prevent browser freeze from large base64 data
                            first_result = query_result.result.data_array[0]
                            if isinstance(first_result, (list, tuple)) and len(first_result) > 0:
                                st.write("**First result (limited):**", [str(item)[:100] + "..." if len(str(item)) > 100 else item for item in first_result])
                            else:
                                result_str = str(first_result)
                                st.write("**First result:**", result_str[:500] + "..." if len(result_str) > 500 else result_str)
                    
                    # Try to display results safely
                    try:
                        results_data = []
                        for i, result in enumerate(query_result.result.data_array):
                            result_dict = {'Rank': i + 1}
                            
                            # Handle different result formats - only extract the columns we want
                            if isinstance(result, dict):
                                # Extract only the columns we want
                                for key in ['file_path', 'file_name', 'folder']:
                                    if key in result:
                                        result_dict[key] = str(result[key])
                            elif isinstance(result, (list, tuple)):
                                # If result is a list/tuple, map to our desired columns only
                                columns = ["file_path", "file_name", "folder"]
                                for j, value in enumerate(result):
                                    if j < len(columns):
                                        col_name = columns[j]
                                        result_dict[col_name] = str(value)
                            else:
                                result_dict['Data'] = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                                
                            results_data.append(result_dict)
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Add image display section
                        if not results_df.empty and 'file_path' in results_df.columns:
                            st.markdown("---")
                            st.subheader("📸 Image Results")
                            
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
                                                    st.write(f"Path: {image_path}")
                                                    st.write(f"Error: {str(img_error)[:100]}...")
                            else:
                                st.info("No image files found in the results")
                        
                    except Exception as display_error:
                        st.error(f"Error displaying results: {display_error}")
                        st.write("**Raw results:**")
                        for i, result in enumerate(query_result.result.data_array[:3]):  # Show first 3 only
                            st.write(f"**Result {i+1}:**", str(result)[:200] + "..." if len(str(result)) > 200 else str(result))
                            
                else:
                    st.warning("No results found in vector search")
                    
            except Exception as e:
                st.error(f"❌ Error during vector search: {e}")
                st.write("**Make sure you have proper permissions for the vector search index**") 