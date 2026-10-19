import numpy as np
import matplotlib.pyplot as plt

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixStatistics,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
    TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)

    if len(matrices) == 0:
        raise ValueError("Набор матриц не должен быть пустым.")

    result = np.zeros_like(
        np.matmul(matrices[0], vectors[0]),
        dtype=np.result_type(matrices, vectors, float),
    )

    for matrix, vector in zip(matrices, vectors):
        result += matrix @ vector

    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix = np.asarray(data.matrix)
    threshold = data.threshold

    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)

    return [
        np.unique(row).tolist()
        for row in matrix
    ]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)

    return [
        np.unique(matrix[:, column]).tolist()
        for column in range(matrix.shape[1])
    ]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns = data.rows, data.columns
    mean, std, seed = data.mean, data.std, data.seed

    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        loc=mean,
        scale=std,
        size=(rows, columns),
    )

    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )

def plot_matrix_statistics(data: MatrixStatistics) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(data.matrix.ravel(), bins=20, edgecolor="black")
    axes[0, 0].set_title("Элементы матрицы")
    axes[0, 0].set_xlabel("Значение")
    axes[0, 0].set_ylabel("Частота")

    axes[0, 1].hist(data.row_means, bins="auto", edgecolor="black")
    axes[0, 1].set_title("Средние по строкам")

    axes[0, 2].hist(data.column_means, bins="auto", edgecolor="black")
    axes[0, 2].set_title("Средние по столбцам")

    axes[1, 0].hist(data.row_variances, bins="auto", edgecolor="black")
    axes[1, 0].set_title("Дисперсии по строкам")

    axes[1, 1].hist(data.column_variances, bins="auto", edgecolor="black")
    axes[1, 1].set_title("Дисперсии по столбцам")

    axes[1, 2].set_visible(False)
    fig.tight_layout()
    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns = data.rows, data.columns
    first, second = data.first, data.second

    row_indices = np.arange(rows).reshape(-1, 1)
    column_indices = np.arange(columns).reshape(1, -1)

    mask = (row_indices + column_indices) % 2 == 0

    return np.where(mask, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color = data.shape_color
    background_color = data.background_color

    image = np.empty(
        (image_height, image_width, 3),
        dtype=np.uint8,
    )
    image[:] = background_color

    if width <= 0 or height <= 0:
        return image

    x_start = (image_width - width) // 2
    y_start = (image_height - height) // 2

    x_end = x_start + width
    y_end = y_start + height

    x_start_clipped = max(0, x_start)
    y_start_clipped = max(0, y_start)
    x_end_clipped = min(image_width, x_end)
    y_end_clipped = min(image_height, y_end)

    if x_start_clipped < x_end_clipped and y_start_clipped < y_end_clipped:
        image[
            y_start_clipped:y_end_clipped,
            x_start_clipped:x_end_clipped,
        ] = shape_color

    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color = data.shape_color
    background_color = data.background_color

    image = np.empty(
        (image_height, image_width, 3),
        dtype=np.uint8,
    )
    image[:] = background_color

    if semi_axis_x <= 0 or semi_axis_y <= 0:
        return image

    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2

    y, x = np.ogrid[:image_height, :image_width]

    ellipse_mask = (
        ((x - x0) / semi_axis_x) ** 2
        + ((y - y0) / semi_axis_y) ** 2
        <= 1
    )

    image[ellipse_mask] = shape_color

    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.asarray(data.values, dtype=float)
    window = data.window

    if values.ndim != 1:
        raise ValueError("Временной ряд должен быть одномерным.")

    if len(values) == 0:
        raise ValueError("Временной ряд не должен быть пустым.")

    if window < 1 or window > len(values):
        raise ValueError("Размер окна должен быть от 1 до длины ряда.")

    local_maxima = np.where(
        (values[1:-1] > values[:-2])
        & (values[1:-1] > values[2:])
    )[0] + 1

    local_minima = np.where(
        (values[1:-1] < values[:-2])
        & (values[1:-1] < values[2:])
    )[0] + 1

    moving_average = np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )

    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)
    class_count = data.class_count

    if labels.ndim != 1:
        raise ValueError("Вектор меток должен быть одномерным.")

    if np.any(labels < 0):
        raise ValueError("Метки классов должны быть неотрицательными.")

    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0

    if class_count < 0:
        raise ValueError("Количество классов не может быть отрицательным.")

    if labels.size and np.any(labels >= class_count):
        raise ValueError("Метка класса выходит за пределы class_count.")

    result = np.zeros((len(labels), class_count), dtype=int)

    if labels.size:
        result[np.arange(len(labels)), labels] = 1

    return result
