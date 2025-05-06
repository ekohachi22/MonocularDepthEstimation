from .base_model import BaseModel
from ..data_handling.data_loading import ImageDepthDataset

import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn import init
from kornia.losses import ssim_loss


import albumentations as A
from albumentations.pytorch import ToTensorV2
import kornia.filters as KF

shared_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ],
    additional_targets={"depth": "image"},
)

input_only_transforms = A.Compose(
    [
        A.RandomGamma(p=0.3),
        A.ColorJitter(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
    ]
)

SAVE_PATH = "last_checkpoint.pth"

def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def depth_smoothness_loss(depth, image):
    dx_depth = gradient_x(depth)
    dy_depth = gradient_y(depth)

    dx_image = gradient_x(image)
    dy_image = gradient_y(image)

    weights_x = torch.exp(-torch.mean(torch.abs(dx_image), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(dy_image), dim=1, keepdim=True))

    smoothness_x = dx_depth * weights_x
    smoothness_y = dy_depth * weights_y

    return smoothness_x.abs().mean() + smoothness_y.abs().mean()

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, y_pred, y):
        return ssim_loss(y_pred, y, window_size=self.window_size, reduction="mean")


class ConvnetModelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()

    def forward(self, y_pred, Y, X):
        ssim_val = self.ssim(y_pred, Y)
        smooth_val = depth_smoothness_loss(y_pred, X)
        l1_val = self.l1(y_pred, Y)

        #print(f"SSIM: {ssim_val.item()}, Smooth: {smooth_val.item()}, L1: {l1_val.item()}")

        return 2 * ssim_val +  smooth_val + 0.001 * l1_val



class ConvnetModel(BaseModel):
    def __init__(self):
        super().__init__()
        self._model = _UNet(6)

    def train(self, X, Y, **kwargs):
        max_iters_no_improvement = 2
        iters_no_improvement = 0
        min_val_loss = float("inf")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_val_dataset = ImageDepthDataset(
            X,
            Y,
            shared_transform=shared_transforms,
            input_only_transform=input_only_transforms,
        )

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
        criterion = ConvnetModelLoss()
        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        for epoch in range(n_epochs):
            self._model.train()
            train_loss = 0.0

            for inputs, targets in tqdm.tqdm(train_loader):
                inputs = inputs.to(device).float()
                targets = targets.to(device).float()

                outputs = self._model(inputs)
                loss = criterion(outputs, targets, inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device).float()
                    targets = targets.to(device).float()

                    outputs = self._model(inputs)
                    loss = criterion(outputs, targets, inputs)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                iters_no_improvement = 0
                torch.save(self._model.state_dict(), SAVE_PATH)

            else:
                iters_no_improvement += 1
                if iters_no_improvement == max_iters_no_improvement:
                    print(
                        f"{iters_no_improvement} iterations reached with no improvement to validation loss! stopping..."
                    )
                    self._model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
                    break

            print(
                f"Epoch [{epoch+1}/{n_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
            )

    def test(self, X, Y):
        return super().test(X, Y)


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, pooling: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu2 = nn.ReLU()

        self.pooling = pooling
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'relu1'], ['conv2', 'relu2']], inplace=True)



class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, encoder_tensor, decoder_tensor):
        decoder_tensor = self.upconv(decoder_tensor)
        x = torch.cat((decoder_tensor, encoder_tensor), 1)
        x = self.dropout(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'relu1'], ['conv2', 'relu2']], inplace=True)



class _UNet(nn.Module):
    def __init__(self, depth: int = 5, start_filters: int = 64, in_channels: int = 3):
        super().__init__()
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        self.depth = depth
        self.start_filters = start_filters
        self.in_channels = in_channels
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

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

        self.reset_params()

    def forward(self, x):
        encoder_outs = []

        x = self.quant(x)

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        x = self.dequant(x)
        return torch.relu(x)
    
    def fuse_model(self):
        for module in self.down_convs:
            if hasattr(module, 'fuse_model'):
                module.fuse_model()
        for module in self.up_convs:
            if hasattr(module, 'fuse_model'):
                module.fuse_model()


    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
