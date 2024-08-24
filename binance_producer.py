import time
import json

from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaProducer
from binance.spot import Spot

topic_name = 'binance'
symbol="BTCUSDT"


producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

def message_handler(message):
    
    print(message)
    data = {
        'open_time': message[0][0],
        'open': message[0][1], 
        'high': message[0][2],  
        'low': message[0][3], 
        'close': message[0][4],
        'volume': message[0][5],
        "close_time": message[0][6],
        "num_trade": message[0][8],
    }

    producer.send(topic_name, value=data)
    producer.flush()


def main():
    while True:
        client = Spot()
        result = client.klines("BTCUSDT", "1m", limit=1)
        message_handler(result)
        time.sleep(60)

if __name__ == "__main__":
    main()