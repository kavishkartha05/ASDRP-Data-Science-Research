from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax 

tweet = "@kkxvish today is a nice day @ home 😊 https://nytimes.com"

# pre-processed tweet
tweet_words = []

for word in tweet.split(''):
    if word.startswith("@") and len(word) > 1:
        word = "@user"
    elif word.startswith("http"):
        word = "http"
    tweet_words.append(word)

print(tweet_words)
