import sys
from collections import defaultdict, Counter
import math
import random
import os
import os.path
from typing import List, Tuple

"""
COMS W4705 - Natural Language Processing - Spring 2025 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

BOS_TOKEN = "START"
EOS_TOKEN = "END"


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, "r") as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence: List[str], n: int) -> List[Tuple]:
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1
    """

    padded_sequence = ["START"] * (n - 1) + sequence + ["STOP"]

    ngrams = []
    for i in range(len(padded_sequence) - n + 1):
        ngram = tuple(padded_sequence[i : i + n])
        ngrams.append(ngram)

    return ngrams


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        self.total_words = sum(self.unigramcounts.values())

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        self.unigramcounts = Counter()
        self.bigramcounts = Counter()
        self.trigramcounts = Counter()

        for sentence in corpus:
            self.unigramcounts.update(get_ngrams(sentence, 1))
            self.bigramcounts.update(get_ngrams(sentence, 2))
            self.trigramcounts.update(get_ngrams(sentence, 3))

    def raw_trigram_probability(self, trigram) -> float:
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if self.total_words > 0:
            count_trigram = self.trigramcounts.get(trigram, 0)
            count_prefix = self.bigramcounts.get((trigram[0], trigram[1]), 0)

            if count_prefix == 0:
                # 1 / |V| if unseen 
                return float(1.0 / len(self.lexicon))
            return count_trigram / count_prefix
        return 0.0

    def raw_bigram_probability(self, bigram) -> float:
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.total_words > 0:
            count_bigram = self.bigramcounts.get(bigram, 0)
            count_prefix = self.unigramcounts.get((bigram[0],), 0)

            if count_prefix == 0:
                return 0.0
            return count_bigram / count_prefix

        return 0.0

    def raw_unigram_probability(self, unigram) -> float:
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        if self.total_words > 0:
            count = self.unigramcounts.get(unigram, 0)
            return float(count / self.total_words)
        return 0.0


    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result

    def smoothed_trigram_probability(self, trigram: Tuple[str, str, str]) -> float:
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).

        The smoothed trigram probability is calculated using linear interpolation:
        P(w | u, v) = λ1 * P_mle(w | u, v) + λ2 * P_mle(w | v) + λ3 * P_mle(w)

        - λ1, λ2, λ3 are interpolation weights (e.g., λ1 = λ2 = λ3 = 1/3)
        - P_mle(w | u, v) is the maximum likelihood estimate (MLE) of the trigram probability
        - P_mle(w | v) is the MLE of the bigram probability
        - P_mle(w) is the MLE of the unigram probability

        """
        lambda1 = 1 / 3.0 # unigram
        lambda2 = 1 / 3.0 # bigram
        lambda3 = 1 / 3.0 # trigram

        u, w, v = trigram

        trigram_prob = self.raw_trigram_probability((u, w, v))
        bigram_prob = self.raw_bigram_probability((w, v))
        unigram_prob = self.raw_unigram_probability((v,))

        smoothed_prob = (
            lambda1 * unigram_prob  +
            lambda2 * bigram_prob +
            lambda3 * trigram_prob
        )

        return smoothed_prob


    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        return float("-inf")

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        return float("inf")


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp1 = model1.perplexity(
            corpus_reader(os.path.join(testdir1, f), model1.lexicon)
        )
        pp2 = model2.perplexity(
            corpus_reader(os.path.join(testdir1, f), model2.lexicon)
        )
        # ..

    for f in os.listdir(testdir2):
        ...
        # ..

    return 0.0


if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)

    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)
