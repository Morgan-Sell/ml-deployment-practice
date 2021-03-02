from my_module import square

import pytest

# each "inputs" value is used in a test.
# Useful to touch edge cases.
@pytest.mark.parametrize(
    'inputs', [
    2, 3, 4]
)
def test_square_return_value_typ_is_int(inputs):
    # When
    subject = square(inputs)

    # Then
    assert isinstance(subject, int)