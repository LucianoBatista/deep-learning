import torch
from model import Net


def run():
    params_model = {
        "input_shape": (3, 256, 256),
        "initial_filters": 16,
        "num_outputs": 2,
    }

    model = Net(params_model)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)

    print(model)


if __name__ == "__main__":
    run()
