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
EOS_TOKEN = "STOP"
UNK_TOKEN = "UNK"


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, "r") as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else UNK_TOKEN for word in sequence]
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

    padded_sequence = [BOS_TOKEN] * (n - 1) + sequence + [EOS_TOKEN]

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
        self.lexicon.add(UNK_TOKEN)
        self.lexicon.add(BOS_TOKEN)
        self.lexicon.add(EOS_TOKEN)

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
        return NotImplementedError
        # return result

    def smoothed_trigram_probability(self, trigram: Tuple[str, str, str]) -> float:
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """

        # same lambda for all n_grams
        n_gram_lambda = 1 / 3.0

        u, w, v = trigram

        trigram_prob = self.raw_trigram_probability((u, w, v))
        bigram_prob = self.raw_bigram_probability((w, v))
        unigram_prob = self.raw_unigram_probability((v,))

        smoothed_prob = (
            n_gram_lambda * unigram_prob
            + n_gram_lambda * bigram_prob
            + n_gram_lambda * trigram_prob
        )

        return smoothed_prob

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)

        log_prob = 0.0

        for trigram in trigrams:
            prob = self.smoothed_trigram_probability(trigram)
            assert prob > 0, "Probability is not greater than 0"

            log_prob += math.log2(prob)

        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """

        running_sum_probability = 0.0  # l
        counts = 0  # M

        for sentence in corpus:
            counts += len(sentence) + 1  # with EOS token
            running_sum_probability += self.sentence_logprob(sentence)

        avg_log_prob = running_sum_probability / counts
        perplexity = 2 ** (-avg_log_prob)

        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    def _get_correctness(model1, model2, testdir, expected_model):
        correct = 0
        total = 0

        for f in os.listdir(testdir):
            filepath = os.path.join(testdir, f)
            pp1 = model1.perplexity(corpus_reader(filepath, model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(filepath, model2.lexicon))

            total += 1
            if (expected_model == 1 and pp1 < pp2) or (
                expected_model == 2 and pp2 < pp1
            ):
                correct += 1

        return correct, total

    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    correct1, total1 = _get_correctness(model1, model2, testdir1, expected_model=1)
    correct2, total2 = _get_correctness(model1, model2, testdir2, expected_model=2)

    total = total1 + total2
    correct = correct1 + correct2

    assert total > 0, "No files in test directories."
    return correct / total


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
