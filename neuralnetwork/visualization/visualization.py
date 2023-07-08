import matplotlib.pyplot as plt
import seaborn as sns

import torch

def plot_prediction(
    y: torch.Tensor, yhat: torch.Tensor, name: str, yname: str = "Ground Truth", error_name: str = "Error"
) -> plt.Figure:
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    for i, (name, data) in enumerate((
        (yname, y),
        (name, yhat),
        (error_name, y - yhat),
    )):
        axs[i].imshow(data, cmap="RdBu")
        axs[i].set_title(name)
        axs[i].axis("off")

    return fig


def error_histogram(
    y: torch.Tensor, yhat: torch.Tensor, ax: plt.Axes = None
) -> tuple[plt.Figure, plt.Axes]:
    error = (y - yhat).flatten()  # (time * lat * lon)
    error = error.detach().numpy()

    ax = ax if ax is not None else plt.gca()
    sns.histplot(error, ax=ax, kde=True, stat="density", common_norm=False)

    return ax


if __name__ == "__main__":

    X = torch.randn(32, 32)
    coef = (torch.rand_like(X) * 2 - 1) * .2
    y = X * coef
    yhat = X * coef + torch.randn_like(X) * 0.1 + 1

    fig = plot_prediction(y, yhat, "Prediction")
    
    fig, ax = plt.subplots()
    error_histogram(y, X, ax = ax)
    error_histogram(y, yhat, ax = ax)
    
    plt.show()
