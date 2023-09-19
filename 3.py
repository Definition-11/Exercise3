import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


with open('Moby_Dick.txt',encoding='gbk', errors='ignore') as file:
    text = file.read()

tokens = word_tokenize(text)

stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

pos_tags = pos_tag(filtered_tokens)

pos_freq = nltk.FreqDist([tag for token, tag in pos_tags])
common_pos = pos_freq.most_common(5)
print("5 most common parts of speech and their frequencies:")
for pos, freq in common_pos:
    print(f"{pos}: {freq}")


lemmatizer = WordNetLemmatizer()
top_20_tokens = nltk.FreqDist([token for token, tag in pos_tags]).most_common(20)
lemmatized_tokens = [lemmatizer.lemmatize(token, tag) for token, tag in pos_tags if (token, tag) in top_20_tokens]


pos_freq = nltk.FreqDist([tag for token, tag in pos_tags])
pos_freq.plot(10, cumulative=False)
plt.show()