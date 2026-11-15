import torch


def pytest_sessionstart(session):
    # Small fixtures otherwise spend more time on thread overhead than convolution.
    torch.set_num_threads(2)
