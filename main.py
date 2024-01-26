from collections import Counter
from nltk.corpus import stopwords
import re
import nltk

nltk.download('stopwords')

with open('rt-polaritydata/rt-polarity.pos', 'r') as file:
    temp_pos_lines = file.readlines()

pos_lines = [line.strip() for line in temp_pos_lines]

with open('rt-polaritydata/rt-polarity.neg', 'r') as file:
    temp_neg_lines = file.readlines()

neg_lines = [line.strip() for line in temp_neg_lines]


def get_top_words(file_path, top_n=1250):  # test function to find the top words
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    words = re.findall(r'\b\w+\b', text.lower())

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    word_counts = Counter(words)
    top_words = word_counts.most_common(top_n)

    return top_words


top_pos_words = get_top_words('rt-polaritydata/rt-polarity.pos')
top_neg_words = get_top_words('rt-polaritydata/rt-polarity.neg')

positive_keywords = []
negative_keywords = []

for word, count in top_pos_words:
    positive_keywords.append(word)

for word, count in top_neg_words:
    negative_keywords.append(word)


def classifier(review):
    # Keywords
    positive_count = sum(keyword in review.lower() for keyword in positive_keywords)
    negative_count = sum(keyword in review.lower() for keyword in negative_keywords)

    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:  # neutral case will return positive
        return 'positive'


pos_total = 0
pos_correct_guess = 0

for line in pos_lines:
    pos_total = pos_total + 1
    result = classifier(line)
    if result == 'positive':
        pos_correct_guess = pos_correct_guess + 1

print("positive accuracy is " + str(pos_correct_guess / pos_total))

neg_total = 0
neg_correct_guess = 0

for line in neg_lines:
    neg_total = neg_total + 1
    result = classifier(line)
    if result == 'negative':
        neg_correct_guess = neg_correct_guess + 1

print("negative accuracy is " + str(neg_correct_guess / neg_total))
