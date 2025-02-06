
import pytest

from trigram_model import get_ngrams

@pytest.mark.parametrize(
    "sequence, n, expected",
    [
        (
            ["natural", "language", "processing"],
            1,
            [('natural',), ('language',), ('processing',), ('STOP',)]
        ),
        (
            ["natural", "language", "processing"],
            2,
            [('START', 'natural'), ('natural', 'language'), ('language', 'processing'), ('processing', 'STOP')]
        ),
        (
            ["natural", "language", "processing"],
            3,
            [('START', 'START', 'natural'), ('START', 'natural', 'language'), ('natural', 'language', 'processing'), ('language', 'processing', 'STOP')]
        )
    ]
)
def test_get_ngrams(sequence, n, expected):
    """Test the get_ngrams function with various inputs."""
    assert get_ngrams(sequence, n) == expected
