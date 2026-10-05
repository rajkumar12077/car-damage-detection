import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CarDamageCNN(nn.Module):
    def __init__(self, num_car_classes: int = 6, num_damage_classes: int = 7):
        super(CarDamageCNN, self).__init__()
        self.num_car_classes = num_car_classes
        self.num_damage_classes = num_damage_classes

        self.block1 = ConvBlock(3, 32, dropout_rate=0.1)
        self.block2 = ConvBlock(32, 64, dropout_rate=0.15)
        self.block3 = ConvBlock(64, 128, dropout_rate=0.2)
        self.block4 = ConvBlock(128, 128, dropout_rate=0.2)
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.shared_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25)
        )

        self.car_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_car_classes)
        )

        self.damage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_damage_classes)
        )

        self.percent_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.feature_maps = None
        self.gradients = None
        self._register_gradcam_hook()

    def _register_gradcam_hook(self):
        final_conv = self.block5[0]

        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        final_conv.register_forward_hook(forward_hook)
        final_conv.register_full_backward_hook(backward_hook)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

    def forward(self, x: torch.Tensor):
        feat = self.forward_features(x)
        pooled = self.global_pool(feat).flatten(1)
        shared = self.shared_fc(pooled)

        car_logits = self.car_head(shared)
        damage_logits = self.damage_head(shared)
        damage_percent = self.percent_head(shared) * 100.0

        return car_logits, damage_logits, damage_percent


def build_model(num_car_classes: int = 6, num_damage_classes: int = 7, device: str = "cpu") -> CarDamageCNN:
    model = CarDamageCNN(num_car_classes=num_car_classes, num_damage_classes=num_damage_classes)
    model.to(device)
    return model


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 160, 160)
    model = build_model(6, 7)
    cars, damages, percents = model(dummy_input)
    print("Model initialized successfully!")
    print(f"Car logits shape: {cars.shape}")
    print(f"Damage logits shape: {damages.shape}")
    print(f"Damage percentage shape: {percents.shape}")
    print(f"Sample damage percentage: {percents.detach().numpy().flatten()}%")
