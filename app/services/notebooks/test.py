
# COMMAND ----------

# Python is the default language, so no magic command is needed.
# Use dbutils for utilities like widgets
dbutils.widgets.text("target_table", "average_prices", "Target Table Name")

# Load a sample dataset into a Spark DataFrame
diamonds = spark.read.csv("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header="true", inferSchema="true")

# Register the DataFrame as a temporary view for SQL access
diamonds.createOrReplaceTempView("diamonds_data")

# Display the first few rows of the DataFrame
display(diamonds.limit(5))

# COMMAND ----------

print("t")
