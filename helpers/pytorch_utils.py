import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("gpu")


def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()

