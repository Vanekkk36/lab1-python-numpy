from numpy_tasks import matrix_statistics, plot_matrix_statistics
from grader_contracts.numpy_tasks import RandomMatrixInput


data = RandomMatrixInput(
    rows=100,
    columns=100,
    mean=0,
    std=1,
    seed=42,
)

statistics = matrix_statistics(data)
plot_matrix_statistics(statistics)
