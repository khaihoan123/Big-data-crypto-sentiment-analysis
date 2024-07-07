import praw
import json

from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaProducer

topic_name =  'reddit'
user_agent = 'test'
client_id = 'SK33APXjmS-pVfsv_APmWg'
client_secret = 'eWUVd7dLhMukZ9zJcpOgF78HT-jL0Q'

#create topic
admin_client = KafkaAdminClient(
    bootstrap_servers="localhost:9092", 
    client_id='test'
)
topics = admin_client.list_topics()
if topic_name not in topics:
    topic_list = []
    topic_list.append(NewTopic(name=topic_name, num_partitions=1, replication_factor=1))
    admin_client.create_topics(new_topics=topic_list, validate_only=False)


#create producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )

producer.send(topic_name, value={"Topic": "Test", "Message": "Test"},)
# producer.send(topic_name, value='ssssss',)
producer.flush()

# reddit = praw.Reddit(
#     client_id =client_id,
#     client_secret =client_secret,
#     user_agent =user_agent,
#     password="@Toilaai123",
#     username="khaihoan123",
# )

# subreddit = reddit.subreddit("CryptoCurrency")

# print(subreddit.display_name)
# print(subreddit.title)

 
# for comment in subreddit.stream.comments():
#     if comment is None:
#         break
#     print('------------------')
#     print(comment.body)
#     # producer.send(topic_name, value=submission.title)
#     # producer.flush()
# for submission in subreddit.stream.submissions():
#     if submission is None:
#         break
#     print(submission.title)
    # producer.send(topic_name, value=submission.title)
    # producer.flush()

# for sub in reddit.subreddit('CryptoCurrency').hot(limit=5):
#     print(sub.title)




