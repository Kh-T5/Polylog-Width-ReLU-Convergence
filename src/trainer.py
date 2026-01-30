import torch
import torch.nn as nn
from tqdm import tqdm
from src.model_utils import ShallowNet, logistic_loss


def get_device():
    """
    Allocating tensors and model to gpu when possible,
    Since I am using Apple Sillicon chips, I call "mps" which is the equivalent to Nvidia's cuda
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_student(
    model: ShallowNet,
    X: torch.Tensor,
    y: torch.Tensor,
    lr=0.1,
    epochs=1000,
    track_flips=True,
):
    """
    Standard Gradient Descent loop using logistic loss.
    Tracks activation flips to validate Lemma 2.2 of Ji & Telgarsky (2019).
    """
    device = get_device()
    model.to(device)
    X, y = X.to(device), y.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if track_flips:
        with torch.no_grad():
            initial_activations = model.get_activations(X)

    history = {"loss": [], "accuracy": [], "flip_ratio": []}

    pbar = tqdm(range(epochs))
    for _ in pbar:
        optimizer.zero_grad()

        # Forward & backward pass
        logits = model(X)
        loss = logistic_loss(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.sign(logits)
            acc = (preds == y).float().mean()

            if track_flips:
                current_activations = model.get_activations(X)
                diff = torch.abs(current_activations - initial_activations)
                flip_ratio = diff.mean().item()
            else:
                flip_ratio = 0.0

        # Save History
        history["loss"].append(loss.item())
        history["accuracy"].append(acc.item())
        history["flip_ratio"].append(flip_ratio)

        pbar.set_description(f"Acc: {acc.item():.4f} | Flips: {flip_ratio:.4f}")

    return history
