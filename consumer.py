import kafka
# import pandas as pd
import os
import time
# os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.2.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 pyspark-shell'

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

topic_name =  'reddit'


def main():
    # consumer = kafka.KafkaConsumer(
    #     topic_name,
    #     group_id='test', 
    #     bootstrap_servers=['localhost:9092'],
    #     value_deserializer=lambda x: x.decode('utf-8'),
    #     enable_auto_commit=True,
    #     auto_offset_reset='earliest',
    #     )
    
    
    
    # if topic_name not in consumer.topics(): 
    #     raise RuntimeError()


    
    # for message in consumer:
    #     data = message.value
    #     print("Received data: ", data)
    kafka_options = {
        "kafka.bootstrap.servers": "172.25.0.4:9092",
        "startingOffsets": "earliest", # Start from the beginning when we consume from kafka
        "subscribe": topic_name,           # Our topic name
        # 'checkpointLocation': '/tmp/checkpoint'
    }
    spark = SparkSession.builder \
        .appName("HelloWorld") \
        .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0') \
        .getOrCreate()

    df = spark \
        .readStream \
        .format("kafka") \
        .options(**kafka_options) \
        .load()
    print('=========')
    # json_schema = StructType([
    #         StructField("Topic", StringType()),
    #         StructField("Message", StringType()),
    #         ])
    df.selectExpr("CAST(value AS STRING)").writeStream.outputMode("append").format("console").start().awaitTermination()
    # data = [("Java", "20000"), ("Python", "100000"), ("Scala", "3000")]
    # df = spark.createDataFrame(data)
    df.show()
if __name__ == "__main__":
    main()