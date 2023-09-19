from textblob import TextBlob

with open('Moby_Dick.txt', 'r', encoding='utf-8') as file:
    text = file.read()

blob = TextBlob(text)

sentiment = blob.sentiment

average_score = sentiment.polarity / 2

if average_score > 0.05:
    overall_sentiment = "positive"
else:
    overall_sentiment = "negative"

print("Average Sentiment Score:", average_score)
print("Overall Text Sentiment:", overall_sentiment)