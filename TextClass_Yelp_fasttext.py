# https://medium.com/@ageitgey/text-classification-is-your-new-secret-weapon-7ca4fad15788
# Install: https://github.com/facebookresearch/fastText/tree/master/python
# https://fasttext.cc/docs/en/supervised-tutorial.html
# Pathlib: http://pbpython.com/pathlib-intro.html

import json
from pathlib import Path
import re
import random
import fastText
from fastText import train_supervised

# Read the Yelp dataset, remove any string formatting and save separate training and test files
path = "C:\\Data\\test_data\\Yelp\\"
reviews_data = Path(path) / "yelp_academic_dataset_review.json"
training_data = Path(path) / "fasttext_dataset_training.txt"
test_data = Path(path) / "fasttext_dataset_test.txt"

percent_test_data = 0.10

def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

with reviews_data.open(encoding="utf8") as input, training_data.open("w", encoding="utf8") as train_output, test_data.open("w", encoding="utf8") as test_output:

    for line in input:
        review_data = json.loads(line)

        rating = review_data['stars']
        text = review_data['text'].replace("\n", " ")
        text = strip_formatting(text)

        fasttext_line = "__label__{} {}".format(rating, text)

        if random.random() <= percent_test_data:
            test_output.write(fasttext_line + "\n")
        else:
            train_output.write(fasttext_line + "\n")


# Python wrapper for fastTex

# Build model
# train_supervised uses the same arguments and defaults as the fastText cli
classifier = train_supervised(input=path + "fasttext_dataset_training.txt", epoch=5, lr=1.0, wordNgrams=2, loss="hs")

# Test model
print(classifier.test(path + "fasttext_dataset_test.txt"))

# Save and load model to/from file
classifier.save_model(path + "reviews_model_ngrams.bin")

# --> Start from here if model was already saved
classifier = fastText.load_model(path + "reviews_model_ngrams.bin")

# Reviews to check
reviews = [
    "This restaurant literally changed my life. This is the best food I've ever eaten!",
    "I hate this place so much. They were mean to me.",
    "I don't know. It was ok, I guess. Not really sure what to say."
]

# Pre-process the text of each review so it matches the training format
preprocessed_reviews = list(map(strip_formatting, reviews))

# Classify review
labels, probabilities = classifier.predict(preprocessed_reviews, 1)

# Print the results
for review, label, probability in zip(reviews, labels, probabilities):
    stars = int(label[0][-1])

    print("{} ({}% confidence)".format("☆" * stars, int(probability[0] * 100)))
    print(review, '\n')


# Combine above steps to a function
def review(reviewtext):
    preprocessed = strip_formatting(reviewtext)
    label, probability = classifier.predict(preprocessed, 1)
    stars = label[0][-1]
    prob = round(probability[0], 4)
    print('Rating:', stars, 'Probability:', prob)
