import yaml


def load_settings():
    with open("settings.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


settings = load_settings()
import torch.nn as nn
import torch
from config import settings

device = torch.device(settings['model']['device'])

class GestureMLP(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 63):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def create_model(num_classes: int = 2):
    """Instantiate the model on the appropriate device."""
    device = torch.device(settings.get('model', {}).get('device', 'cuda'))
    return GestureMLP(num_classes=num_classes).to(device)

model = create_model(num_classes=2)  # will be reloaded when checkpoint changes
model.eval()