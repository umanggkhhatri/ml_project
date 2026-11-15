"""Small U-Net and attention-gated U-Net with arbitrary spatial-size support."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(1, outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(1, outgoing), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class AttentionGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = max(1, width // 2)
        self.skip = nn.Conv2d(width, hidden, 1)
        self.gate = nn.Conv2d(width, hidden, 1)
        self.score = nn.Conv2d(hidden, 1, 1)

    def forward(self, skip, gate):
        weights = torch.sigmoid(self.score(F.relu(self.skip(skip) + self.gate(gate))))
        return skip * weights


class UNet(nn.Module):
    def __init__(self, classes=3, widths=(16, 32, 64, 128, 256), attention=False):
        super().__init__()
        self.encoders = nn.ModuleList()
        incoming = 1
        for width in widths:
            self.encoders.append(ConvBlock(incoming, width))
            incoming = width
        self.up = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.gates = nn.ModuleList()
        for incoming, outgoing in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.up.append(nn.Conv2d(incoming, outgoing, 1))
            self.decoders.append(ConvBlock(outgoing * 2, outgoing))
            self.gates.append(AttentionGate(outgoing) if attention else nn.Identity())
        self.attention = attention
        self.output = nn.Conv2d(widths[0], classes, 1)

    def forward(self, x):
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, decoder, gate, skip in zip(self.up, self.decoders, self.gates, reversed(skips)):
            x = up(F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False))
            if self.attention:
                skip = gate(skip, x)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.output(x)
