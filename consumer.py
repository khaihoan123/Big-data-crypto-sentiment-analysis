from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


kafka_options = {
        "kafka.bootstrap.servers": "localhost:9092",
        "startingOffsets": "latest", # Start from the beginning when we consume from kafka
        "subscribe": 'reddit',           # Our topic name
    }

kafka_options_2 = {
        "kafka.bootstrap.servers": "localhost:9092",
        "startingOffsets": "latest", # Start from the beginning when we consume from kafka
        "subscribe": 'binance',           # Our topic name
    }

schema = StructType([
        StructField("id", StringType()),
        StructField("name", StringType()),
        StructField("author", StringType()),
        StructField("body", StringType()),
        StructField("subreddit_id", StringType()),
        StructField("subreddit", StringType()),
        StructField("upvotes", IntegerType()),
        StructField("downvotes", IntegerType()),
        StructField("over_18", StringType()),
        StructField("created_utc", DoubleType()),
        StructField("permalink", StringType()),
        StructField("submission_id", StringType()),
        StructField("submission_title", StringType()),
        ])

schema_2 = StructType([
        StructField("open_time", LongType()),
        StructField("open", StringType()),
        StructField("high", StringType()),
        StructField("low", StringType()),
        StructField("close", StringType()),
        StructField("volume", StringType()),
        StructField("close_time", LongType()),
        StructField("num_trade", IntegerType()),
        ])

def main():
    
    spark = SparkSession.builder \
        .appName("HelloWorld") \
        .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.mongodb.spark:mongo-spark-connector_2.12:10.3.0') \
        .getOrCreate()

    spark \
        .readStream \
        .format("kafka") \
        .options(**kafka_options) \
        .load() \
        .selectExpr("CAST(value AS STRING)") \
        .select(from_json("value", schema).alias("data")) \
        .select("data.*") \
        .writeStream \
        .format("mongodb")\
        .option("checkpointLocation", "/tmp/pyspark/") \
        .option("forceDeleteTempCheckpointLocation", "true")\
        .option("spark.mongodb.database", 'crypto')\
        .option("spark.mongodb.collection", 'reddit')\
        .outputMode("append").start()

    spark \
        .readStream \
        .format("kafka") \
        .options(**kafka_options_2) \
        .load() \
        .selectExpr("CAST(value AS STRING)") \
        .select(from_json("value", schema_2).alias("data")) \
        .select("data.*") \
        .writeStream \
        .format("mongodb")\
        .option("checkpointLocation", "/tmp/pyspark2/") \
        .option("forceDeleteTempCheckpointLocation", "true")\
        .option("spark.mongodb.database", 'crypto')\
        .option("spark.mongodb.collection", 'btcusdt')\
        .outputMode("append").start()

    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()