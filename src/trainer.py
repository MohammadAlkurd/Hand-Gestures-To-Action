import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from config import create_model, GestureMLP, device, settings
from src.dataset_loader import load_dataset


class GestureLandmarkDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx].clone()
        if self.augment:
            noise = torch.randn_like(features) * 0.01  # small jitter
            features = features + noise
        return features, self.y[idx]


def train_from_cache(cache_root=None, num_epochs=30):
    """Train the gesture classifier from cached landmarks recorded via the UI."""
    from src.dataset_loader import load_landmark_cache
    X, y, label_map = load_landmark_cache(cache_root)
    if len(X) < 4 or len(label_map) < 2:
        raise ValueError("Not enough cached samples to train (record at least 2 gestures first).")

    n = len(X)
    indices = torch.randperm(n)
    split = max(1, min(n - 1, int(0.85 * n))) if n > 1 else 1
    train_idx, val_idx = indices[:split], indices[split:] if split < n else indices[:1]

    train_dataset = GestureLandmarkDataset(X, y, augment=True)
    val_dataset = GestureLandmarkDataset(X, y, augment=False)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=min(32, len(train_idx)), shuffle=True)
    val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=min(32, len(val_idx)), shuffle=False)

    model = create_model(num_classes=len(label_map))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()

    torch.save({'model_state_dict': model.state_dict(), 'label_map': label_map},
               settings['model']['checkpoint_path'])
    return label_map


def train_model(root_dir, num_epochs=50):
    """Train gesture classifier from video dataset."""
    X, y, label_map = load_dataset(root_dir)

    n = len(X)
    indices = torch.randperm(n)
    split = int(0.85 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = GestureLandmarkDataset(X, y, augment=True)
    val_dataset = GestureLandmarkDataset(X, y, augment=False)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=32, shuffle=False)

    model = create_model(num_classes=len(label_map))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += features.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += features.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss/train_total:.4f} | "
              f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_map': label_map,
    }, settings['model']['checkpoint_path'])


def evaluate_model(root_dir):
    """Load checkpoint and evaluate on videos."""
    checkpoint = torch.load(settings['model']['checkpoint_path'], map_location=device)
    num_classes = len(checkpoint['label_map'])
    model = GestureMLP(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    loader = load_dataset(root_dir)
    total_correct, total_samples = 0, 0
    for features, label in loader:
        features = features.to(device).unsqueeze(0)
        with torch.no_grad():
            output = model(features)
            pred = output.argmax(1).item()
            if pred == label:
                total_correct += 1
        total_samples += 1

    return total_correct / total_samples if total_samples > 0 else 0.0