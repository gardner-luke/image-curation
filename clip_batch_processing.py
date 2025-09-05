# Simple CLIP Batch Processing for Delta Tables

import mlflow
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType

# Configuration - Update these values
MODEL_NAME = "autobricks.agriculture.clip_embedding-356"
SOURCE_TABLE = "your_catalog.schema.your_table"
INPUT_COLUMN = "text_column"
INPUT_TYPE = "text"  # or "image"
OUTPUT_TABLE = "your_catalog.schema.output_table"

# Initialize
spark = SparkSession.builder.appName("CLIPBatch").getOrCreate()

# Load model from Unity Catalog
model_uri = f"models:/{MODEL_NAME}/1"
model = mlflow.pyfunc.load_model(model_uri)
print(f"Loaded model: {MODEL_NAME}")

# Create UDF
def get_embedding(input_text):
    if input_text is None:
        return None
    
    input_data = pd.DataFrame({"input_data": [input_text]})
    params = {"input_type": INPUT_TYPE}
    
    try:
        result = model.predict(input_data, params=params)
        return result[0]
    except:
        return None

embedding_udf = udf(get_embedding, ArrayType(FloatType()))

# Process table
df = spark.table(SOURCE_TABLE)
print(f"Processing {df.count()} rows")

result_df = df.withColumn("embeddings", embedding_udf(col(INPUT_COLUMN)))

# Save to Delta table
result_df.write \
         .format("delta") \
         .mode("overwrite") \
         .saveAsTable(OUTPUT_TABLE)

print(f"Results saved to {OUTPUT_TABLE}")

# Show sample
spark.table(OUTPUT_TABLE).select("*").limit(3).show()