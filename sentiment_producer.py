import praw
import json

from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaProducer

topic_name = 'reddit'
reddit_config = {
    'user_agent': 'test',
    'client_id': 'SK33APXjmS-pVfsv_APmWg',
    'client_secret': 'eWUVd7dLhMukZ9zJcpOgF78HT-jL0Q',
    'password': "@Toilaai123",
    'username': "khaihoan123"
}

subreddit_list = 'CryptoCurrency+CryptoMoonShots+CryptoMarkets+CryptocurrencyICO'

# create topic
admin_client = KafkaAdminClient(
    bootstrap_servers="localhost:9092",
    client_id='test'
)
topics = admin_client.list_topics()
if topic_name not in topics:
    topic_list = []
    topic_list.append(
        NewTopic(name=topic_name, num_partitions=1, replication_factor=1))
    admin_client.create_topics(new_topics=topic_list, validate_only=False)


# create producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# producer.send(topic_name, value={"Topic": "b", "Message": "Test"},)
# producer.flush()

reddit = praw.Reddit(**reddit_config)

subreddit = reddit.subreddit(subreddit_list)


for comment in subreddit.stream.comments(skip_existing=True):
    if comment is None:
        break
    print('------------------')
    comment_json = {
        "id": comment.id,
        "name": comment.name,
        "author": comment.author.name,
        "body": comment.body,
        "subreddit_id": comment.subreddit_id,
        "subreddit": comment.subreddit.display_name,
        "upvotes": comment.ups,
        "downvotes": comment.downs,
        "over_18": comment.over_18,
        "created_utc": comment.created_utc,
        "permalink": comment.permalink,
        "submission_id": comment.submission.id,
        "submission_title": comment.submission.title,
    }
    print(comment_json)
    producer.send(topic_name, value=comment_json)
    producer.flush()

# for sub in reddit.subreddit('CryptoCurrency').hot(limit=None):
#     print(sub.title)