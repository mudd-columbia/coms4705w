"""
Consider the following training corpus of emails with the class labels ham and spam. The content of each email has already been processed and is provided as a bag of words.

Email1 (spam): buy car Nigeria profit
Email2 (spam): money profit home bank
Email3 (spam): Nigeria bank check wire
Email4 (ham): money bank home car
Email5 (ham): home Nigeria fly

a) Based on this data, estimate the prior probability for a random email to be spam or ham if we don't know anything about its content, i.e. P(Class)?

b) Based on this data, estimate the conditional probability distributions for each word given the class, i.e. P(Word | Class). You can write down these distributions in a table.

c) Using the Naive Bayes' approach and your probability estimates, what is the predicted class label for each of the following emails?

Nigeria
Nigeria home
home bank money
(Note: Nothing in this exercise is meant to imply that Nigerians are particularly likely to engage in spamming activity. The "prince of Nigeria" scam is a well-known form of fraud, and while Nigeria is the country most often mentioned in this type of scam, the scammers are usually located elsewhere)
"""

from loguru import logger

emails = [
    [("buy", "car", "Nigeria", "profit"), "spam"],
    [("money", "profit", "home", "bank"), "spam"],
    [("Nigeria", "bank", "check", "wire"), "spam"],
    [("money", "bank", "home", "car"), "ham"],
    [("home", "Nigeria", "fly"), "ham"],
]

# problem 1
"""
3 spam, 
2 ham
"""
from collections import Counter, defaultdict

# Problem 1: Calculate prior probabilities P(Class)
label_counts = Counter()
unique_words = set()
words_in_label = defaultdict(Counter)

for email in emails:
    words, label = email
    label_counts[label] += 1
    words_in_label[label].update(words)
    unique_words.update(words)

logger = logger.patch(lambda r: r.update(name="problem 1"))

n_emails = len(emails)
priors = {label: count / n_emails for label, count in label_counts.items()}

logger.info(f"Priors are: {priors}")

print("#################################################")

"""
using 'profit' as an example...

'profit' appeats in 2 of 3 spam emails
P(profit | spam) = 2 / 3 = 0.667

'profit' does nto apprear in ham emails
P(profit | ham) = 0 / 2 = 0

priors:
p(spam) = 0.6
p(ham) = 0.4


"""
word_probs = defaultdict(dict)

# Loop through each class to calculate word probabilities based on email occurrences
for label in words_in_label:
    total_emails_in_class = label_counts[label]  # Number of emails for this class
    for word in unique_words:
        word_probs[label][word] = words_in_label[label][word] / total_emails_in_class

# Log word probabilities for each class
for word in sorted(unique_words):
    probs = {label: word_probs[label][word] for label in words_in_label}
    prob_strs = [f"P({word}|{label}) = {prob:.3f}" for label, prob in probs.items()]
    logger.info(", ".join(prob_strs))

print("#################################################")

# problem 3

"""
For 'Nigeria'
p(spam | Nigeria) ∝ p(spam) * p(Nigieria | spam) = 0.6 * 2/3
= 0.4

p(ham | Nigeria) ∝ p(spam) * p(Nigieria | ham) = 0.4 * 1/2 
= 0.2

0.4 > 0.2 -> spam

For 'Nigeria', 'home' 
P(spam | Nigeria, home) ∝ P(spam) * P(Nigeria|spam) * P(home|spam) # because bayes independence
= 0.6 * (2/3) * (1/3)
= 0.133

P(ham | Nigeria,home) ∝ P(ham) * P(Nigeria|ham) * P(home|ham)
= 0.4 * (1/2) * (2/2)
= 0.2

0.2 > 0.13 -> ham
"""


# Problem 3: Predict class labels for test emails
test_emails = [["Nigeria"], ["Nigeria", "home"], ["home", "bank", "money"]]


def calculate_score(email, label):
    score = priors[label]  # Start with the prior probability P(Class)
    for word in email:
        # If a word does not appear in the training data for this class, it is ignored (score unaffected)
        if word in word_probs[label]:
            score *= word_probs[label][word]
        else:
            score *= 0  # Explicitly multiply by 0 to handle cases where word is missing in class
    return score


# Classify each test email
for email in test_emails:
    scores = {label: calculate_score(email, label) for label in priors}
    predicted = max(scores, key=scores.get)

    # Log the results
    logger.info(f"Email: {' '.join(email)}")
    for label, score in scores.items():
        logger.info(f"P({label}|words) ∝ {score:.6f}")
    logger.info(f"Predicted: {predicted}\n")
