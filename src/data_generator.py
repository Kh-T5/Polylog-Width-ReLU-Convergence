import torch


def generate_teacher_student_data(teacher, n_samples, d, margin_filter=0.1):
    """
    Generates synthetic data (X, y) labeled by a Teacher network.

    Inputs:
        teacher: Teacher class used to generate data.
        n_samples: Number of points.
        d: Input dimension.
        margin_filter: Minimum distance from the decision boundary to
                       ensure the Assumption 2.1 of the paper is well-posed.
    """
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

    return X, y
