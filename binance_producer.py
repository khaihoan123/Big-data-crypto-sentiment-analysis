import time
import json

from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaProducer
from datetime import datetime
from binance.spot import Spot

# coin_list = ['BNBUSDT', 'ETHUSDT', 'DOGEUSDT', 'XRPUSDT' , 'ADAUSDT', 'ARBUSDT', 'MATICUSDT', 'SOLUSDT']
topic_name = 'binance'
symbol="BTCUSDT"


producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

def message_handler(message):
    
    print(message)
    data = {
        'x': message[0][0],
        'o': message[0][1], 
        'h': message[0][2],  
        'l': message[0][3], 
        'c': message[0][4],
        'volume': message[0][5],
        "closetime": message[0][6],
        "num_trade": message[0][8],
    }
        # print(type(event_time))
        # print(datetime.fromtimestamp(event_time / 1000))
    # print(type(message))
    producer.send(topic_name, value=data)
    producer.flush()

# my_client = SpotWebsocketStreamClient(on_message=message_handler)

def main():
    

    # Subscribe to a single symbol stream
    # my_client.kline(symbol=symbol, interval='1m')
        
    # time.sleep(10)
    # my_client.stop()s
    while True:
        client = Spot()
        result = client.klines("BTCUSDT", "1m", limit=1)
        message_handler(result)
        time.sleep(60)

if __name__ == "__main__":
    main()