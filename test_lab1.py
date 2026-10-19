
import numpy as np

from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,
)
from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)
from grader_contracts.python_basics import (
    TextInput,
    PositiveIntegerInput,
    VectorPairInput,
)
from grader_contracts.numpy_tasks import (
    MatrixVectorBatchInput,
    BinarizeInput,
    MatrixInput,
    RandomMatrixInput,
    ChessInput,
    RectangleInput,
    EllipseInput,
    TimeSeriesInput,
    OneHotInput,
)


def test_python_basics():
    assert count_vowels(TextInput("Hello")) == 2
    assert count_vowels(TextInput("AEIOU")) == 5
    assert count_vowels(TextInput("xyz")) == 0

    assert has_unique_characters(TextInput("abc")) is True
    assert has_unique_characters(TextInput("hello")) is False
    assert has_unique_characters(TextInput("")) is True

    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(15)) == 4
    assert count_one_bits(PositiveIntegerInput(8)) == 1

    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4

    assert mse(VectorPairInput([1, 2, 3], [1, 3, 5])) == 5 / 3
    assert mse(VectorPairInput([1, 2], [1, 2])) == 0.0

    assert prime_factorization(PositiveIntegerInput(86240)) == (
        "(2**5)(5)(7**2)(11)"
    )
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"

    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(10)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(1)) == 1

    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    assert is_balanced_number(PositiveIntegerInput(121)) is True
    assert is_balanced_number(PositiveIntegerInput(11)) is True


def test_numpy_tasks():
    batch = MatrixVectorBatchInput(
        matrices=np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]]),
        vectors=np.array([[[1], [2]], [[3], [4]]]),
    )
    np.testing.assert_array_equal(
        sum_prod(batch),
        np.array([[7.0], [10.0]]),
    )

    original = np.array([[0.2, 0.5], [0.7, 0.9]])
    result = binarize(BinarizeInput(original, 0.5))
    np.testing.assert_array_equal(result, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(original, [[0.2, 0.5], [0.7, 0.9]])

    matrix = MatrixInput(np.array([[1, 2, 1], [3, 2, 3]]))
    assert unique_rows(matrix) == [[1, 2], [2, 3]]
    assert unique_columns(matrix) == [[1, 3], [2], [1, 3]]

    stats = matrix_statistics(RandomMatrixInput(3, 4, seed=42))
    assert stats.matrix.shape == (3, 4)
    assert stats.row_means.shape == (3,)
    assert stats.column_means.shape == (4,)
    assert stats.row_variances.shape == (3,)
    assert stats.column_variances.shape == (4,)
    np.testing.assert_allclose(stats.row_means, stats.matrix.mean(axis=1))
    np.testing.assert_allclose(stats.column_means, stats.matrix.mean(axis=0))
    np.testing.assert_allclose(stats.row_variances, stats.matrix.var(axis=1))
    np.testing.assert_allclose(stats.column_variances, stats.matrix.var(axis=0))
    np.testing.assert_array_equal(
        stats.matrix,
        matrix_statistics(RandomMatrixInput(3, 4, seed=42)).matrix,
    )

    board = chess(ChessInput(3, 4, 0, 1))
    np.testing.assert_array_equal(
        board,
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
    )

    rect = draw_rectangle(
        RectangleInput(2, 2, 5, 5, (255, 0, 0), (0, 0, 0))
    )
    assert rect.shape == (5, 5, 3)
    assert np.sum(np.all(rect == [255, 0, 0], axis=2)) == 4

    ellipse = draw_ellipse(
        EllipseInput(2, 1, 5, 5, (255, 0, 0), (0, 0, 0))
    )
    assert ellipse.shape == (5, 5, 3)
    assert np.all(ellipse[2, 2] == [255, 0, 0])

    series = analyze_time_series(TimeSeriesInput([1, 3, 2, 4, 1], 3))
    assert series.mean == 2.2
    assert series.local_maxima_indices.tolist() == [1, 3]
    assert series.local_minima_indices.tolist() == [2]
    np.testing.assert_allclose(
        series.moving_average,
        [2.0, 3.0, 7 / 3],
    )

    encoded = one_hot(OneHotInput([0, 2, 3, 0]))
    np.testing.assert_array_equal(
        encoded,
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ],
    )

    encoded_fixed = one_hot(OneHotInput([0, 1], class_count=4))
    assert encoded_fixed.shape == (2, 4)

def test_python_edge_cases():
    assert count_vowels(TextInput("")) == 0
    assert count_vowels(TextInput("AEIOUaeiou")) == 10
    assert has_unique_characters(TextInput("aA")) is True

    assert count_one_bits(PositiveIntegerInput(255)) == 8
    assert count_one_bits(PositiveIntegerInput(16)) == 1

    assert multiplicative_persistence(PositiveIntegerInput(25)) == 2

    assert mse(VectorPairInput([5], [2])) == 9.0
    assert mse(VectorPairInput([7, 7], [7, 7])) == 0.0

    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"
    assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"

    assert pyramid(PositiveIntegerInput(5)) == 2
    assert pyramid(PositiveIntegerInput(30)) == 4

    assert is_balanced_number(PositiveIntegerInput(7)) is True
    assert is_balanced_number(PositiveIntegerInput(12321)) is True
    assert is_balanced_number(PositiveIntegerInput(1234)) is False

def test_numpy_edge_cases():
    matrix = np.array([[0.5, 0.5001, -1.0]])
    result = binarize(BinarizeInput(matrix, 0.5))

    np.testing.assert_array_equal(result, [[0, 1, 0]])
    np.testing.assert_array_equal(matrix, [[0.5, 0.5001, -1.0]])

    data = MatrixInput(np.array([[5], [5], [7]]))
    assert unique_rows(data) == [[5], [5], [7]]
    assert unique_columns(data) == [[5, 7]]

    stats = matrix_statistics(RandomMatrixInput(1, 1, seed=42))
    assert stats.matrix.shape == (1, 1)
    np.testing.assert_allclose(stats.row_variances, [0.0])
    np.testing.assert_allclose(stats.column_variances, [0.0])

    board = chess(ChessInput(1, 1, 5, 9))
    np.testing.assert_array_equal(board, [[5]])

    series = analyze_time_series(TimeSeriesInput([1, 2, 3], 1))
    np.testing.assert_allclose(series.moving_average, [1, 2, 3])
    assert series.local_maxima_indices.size == 0
    assert series.local_minima_indices.size == 0

    encoded = one_hot(OneHotInput([0, 1], class_count=3))
    np.testing.assert_array_equal(
        encoded,
        [[1, 0, 0], [0, 1, 0]],
    )
