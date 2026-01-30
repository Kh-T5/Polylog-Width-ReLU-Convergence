import torch
from src.model_utils import Teacher


def generate_teacher_student_data(n_samples, d, k, margin_filter=0.1):
    """
    Generates synthetic data (X, y) labeled by a Teacher network.

    Inputs:
        n_samples: Number of points.
        d: Input dimension.
        k: Teacher width.
        margin_filter: Minimum distance from the decision boundary to
                       ensure the Assumption 2.1 of the paper is well-posed.
    """
    teacher = Teacher(d, k)
    X_list = []
    y_list = []

    while len(X_list) < n_samples:
        x = torch.randn(1, d)
        logit = teacher(x)

        if torch.abs(logit) > margin_filter:
            X_list.append(x)
            y_list.append(torch.sign(logit))

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0).reshape(-1, 1)

    return X, y, teacher
