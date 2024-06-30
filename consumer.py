import kafka
import pandas as pd

topic_name =  'twitter'

consumer = kafka.KafkaConsumer(
    topic_name,
    group_id='test', 
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: x.decode('utf-8'),
    enable_auto_commit=True,
    auto_offset_reset='earliest',
    )
if 'twitter' not in consumer.topics(): 
    raise RuntimeError()


for message in consumer:
    data = message.value
    print("Received data: ", data)
    