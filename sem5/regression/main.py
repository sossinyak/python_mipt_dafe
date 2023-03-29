import matplotlib.pyplot as plt
import numpy as np

from computing_core.nonparametric_regression import NonparametricRegression
from computing_core.nonparametric_regression import ComutitionTypes


plt.style.use('ggplot')


if __name__ == '__main__':

    x = np.linspace(-10, 10, 1000)
    y = np.sin(x) +  np.random.normal(size=x.size)

    regressor = NonparametricRegression(
        k_neighbours=100,
        comp_type=ComutitionTypes.NUMPY
    )
    regressor.fit(x, y)

    predictions = regressor.predict(x)

    _, ax = plt.subplots(figsize=(16, 9))

    ax.scatter(x, y, label='source', c='royalblue', s=1)
    ax.plot(x, predictions, label='prediction', c='steelblue')

    ax.legend()

    plt.show()

