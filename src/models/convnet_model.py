from .base_model import BaseModel
from ..data_handling.data_loading import ImageDepthDataset

import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


class ConvnetModel(BaseModel):
    def __init__(self):
        super().__init__()
        self._model = _UNet(3)

    def train(self, X, Y, **kwargs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_val_dataset = ImageDepthDataset(X, Y)

        val_ratio = 0.2
        val_size = int(len(train_val_dataset) * val_ratio)
        train_size = len(train_val_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_val_dataset, [train_size, val_size]
        )

        n_epochs = kwargs["n_epochs"]
        lr = kwargs["lr"]

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        self._model = self._model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            self._model.train()
            train_loss = 0.0

            for inputs, targets in tqdm.tqdm(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = self._model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = self._model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            print(
                f"Epoch [{epoch+1}/{n_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
            )

    def test(self, X, Y):
        return super().test(X, Y)


class _EncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, padding=1, pooling: bool = True
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )

        self.pooling = pooling
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.conv1 = nn.Conv2d(
            2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, encoder_tensor, decoder_tensor):
        decoder_tensor = self.upconv(decoder_tensor)
        x = torch.cat((decoder_tensor, encoder_tensor), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class _UNet(nn.Module):
    def __init__(self, depth: int = 5, start_filters: int = 64, in_channels: int = 3):
        super().__init__()
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        self.depth = depth
        self.start_filters = start_filters
        self.in_channels = in_channels

        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filters * (2**i)
            pooling = True if i < depth - 1 else False

            down_conv = _EncoderBlock(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = _DecoderBlock(ins, outs)
            self.up_convs.append(up_conv)

        self.conv_final = nn.Conv2d(outs, 1, kernel_size=1)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x):
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        return F.sigmoid(x)
