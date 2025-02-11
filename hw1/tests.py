import pytest
import os
from trigram_model import get_ngrams, TrigramModel


@pytest.mark.parametrize(
    "sequence, n, expected",
    [
        (
            ["natural", "language", "processing"],
            1,
            [("natural",), ("language",), ("processing",), ("STOP",)],
        ),
        (
            ["natural", "language", "processing"],
            2,
            [
                ("START", "natural"),
                ("natural", "language"),
                ("language", "processing"),
                ("processing", "STOP"),
            ],
        ),
        (
            ["natural", "language", "processing"],
            3,
            [
                ("START", "START", "natural"),
                ("START", "natural", "language"),
                ("natural", "language", "processing"),
                ("language", "processing", "STOP"),
            ],
        ),
    ],
)
def test_get_ngrams(sequence, n, expected):
    """Test the get_ngrams function with various inputs."""
    assert get_ngrams(sequence, n) == expected


def test_count_ngrams():

    corpus_path = os.path.abspath(os.path.join("hw1", "hw1_data", "brown_train.txt"))

    model = TrigramModel(corpus_path)

    assert model.unigramcounts[("the",)] == 61428
    assert model.bigramcounts[("START", "the")] == 5478
    assert model.trigramcounts[("START", "START", "the")] == 5478

    
