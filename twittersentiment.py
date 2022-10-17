from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax 

tweet = input("Please enter the etx of a tweet to be analyzed with roBERTa: ")

# pre-processed tweet
tweet_words = []

for word in tweet.split(" "):
    if word.startswith("@") and len(word) > 1:
        word = "@user"
    elif word.startswith("http"):
        word = "http"
    tweet_words.append(word)

tweet_processed = " ".join(tweet_words)
print(tweet_processed)

# initiate roBERTa 
roBERTa = "cardiffnlp/twitter-roberta-base-sentiment" 
model = AutoModelForSequenceClassification.from_pretrained(roBERTa) 
tokenizer = AutoTokenizer.from_pretrained(roBERTa)
labels = ["negative", "neutral", "positive"]

# sentiment analysis 
encoded_tweet = tokenizer(tweet_processed, return_tensors = "pt")
output = model(**encoded_tweet)
print(output)
scores = output[0][0].detach().numpy()
print(scores)
scores = softmax(scores)
print(scores)

for i in range(len(scores)):
    l = labels[i]
    s = scores[i]
    print(l + ": " + str(s))
