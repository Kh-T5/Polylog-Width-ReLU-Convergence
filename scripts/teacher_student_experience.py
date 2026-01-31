import torch
import matplotlib.pyplot as plt
from src.model_utils import ShallowNet
from src.data_generator import generate_teacher_student_data
from src.trainer import train_student
from src.model_utils import Teacher


# Config
D = 50
K = 2
N_SAMPLES = 2000
EPOCHS = 5000
LR = 0.1
WIDTHS = [2**i for i in range(2, 13)]
MARGIN_FILTER = 0.3


def run_experiment():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    teacher = Teacher(D, K)
    # Train data
    X, y = generate_teacher_student_data(teacher, N_SAMPLES, D, MARGIN_FILTER)

    # Test data
    X_test, y_test = generate_teacher_student_data(teacher, 200, D, MARGIN_FILTER)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    print(f"Current used device: {device}")
    results = {"width": [], "test_acc": [], "final_flips": []}

    print(f"Starting Width Sweep: {WIDTHS}")

    for m in WIDTHS:
        print(f"\nTraining Student with Width m={m}...")

        student = ShallowNet(D, m)

        history = train_student(student, X, y, lr=LR, epochs=EPOCHS)

        with torch.no_grad():
            test_logits = student(X_test)
            test_acc = (torch.sign(test_logits) == y_test).float().mean().item()

        results["width"].append(m)
        results["test_acc"].append(test_acc)
        results["final_flips"].append(history["flip_ratio"][-1])

    return results


def plot_results(results):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Student Width (m)")
    ax1.set_ylabel("Test Accuracy", color="tab:blue")
    ax1.semilogx(
        results["width"],
        results["test_acc"],
        "o-",
        color="tab:blue",
        label="Test Accuracy",
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", ls="-", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Activation Flip Ratio", color="tab:red")
    ax2.semilogx(
        results["width"],
        results["final_flips"],
        "s--",
        color="tab:red",
        label="Flip Ratio",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Validation of Polylogarithmic Width Sufficiency")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = run_experiment()
    plot_results(data)
