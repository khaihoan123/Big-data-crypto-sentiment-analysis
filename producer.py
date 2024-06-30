import praw

from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaProducer


# topic_name =  'twitter'
# APP_KEY = '1805943490393632768HoanHoang85'
# APP_SECRET = 'YOUR_APP_SECRET'

# # bearer_token = ""
# consumer_key = "ac8bbPnJ8alng97mBB3eZ7r74"
# consumer_secret = "85hQ7GOBKOqKW0ciNoMfQ4Sdc61vUgIXmZO7KqxVShjFARal40"
# access_token = "1805925934437969921-KmLqClo0IN2IH8jyYzw30tvgy54z43"
# access_token_secret = "vVEuWDFUCPA12LDMgvGtOb08rtT6oZT1wRpvDBVfJ03Gw"

# # You can authenticate as your app with just your bearer token
# # client = tweepy.Client(bearer_token=bearer_token)

# # You can provide the consumer key and secret with the access token and access
# # token secret to authenticate as a user
# client = tweepy.Client(
#     consumer_key=consumer_key, consumer_secret=consumer_secret,
#     access_token=access_token, access_token_secret=access_token_secret
# )

# response = client.search_recent_tweets("Tweepy", user_auth=True)

# print(response.meta)

user_agent = 'test'
client_id = 'SK33APXjmS-pVfsv_APmWg'
client_secret = 'eWUVd7dLhMukZ9zJcpOgF78HT-jL0Q'
reddit = praw.Reddit(
    client_id =client_id,
    client_secret =client_secret,
    user_agent =user_agent,
    password="@Toilaai123",
    username="khaihoan123",
)
print(reddit.read_only)
subreddit = reddit.subreddit("redditdev")

print(subreddit.display_name)
print(subreddit.title)
# Output: reddit development
# for sub in reddit.subreddit('CryptoCurrency').hot(limit=5):
#     print(sub.title)

# #create topic
# admin_client = KafkaAdminClient(
#     bootstrap_servers="localhost:9092", 
#     client_id='test'
# )
# topics = admin_client.list_topics()
# if 'twitter' not in topics:
#     topic_list = []
#     topic_list.append(NewTopic(name=topic_name, num_partitions=1, replication_factor=1))
#     admin_client.create_topics(new_topics=topic_list, validate_only=False)


# #create producer
# producer = KafkaProducer(
#     bootstrap_servers='localhost:9092',
#     value_serializer=lambda x: x.encode('utf-8') 
#     )

# producer.send(topic_name, value="Hello, World!")
# producer.flush()
