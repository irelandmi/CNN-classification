import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
import mlflow


def unpickle(file):
	with open(file, 'rb') as fo:
		return pickle.load(fo, encoding='bytes')


def load_cifar10(data_dir=os.path.join(os.path.dirname(__file__), '..', 'datasets', 'cifar-10-batches-py')):
	"""Load all CIFAR-10 batches and return train/test tensors."""
	# Load training data
	train_data = []
	train_labels = []
	for i in range(1, 6):
		batch = unpickle(f'{data_dir}/data_batch_{i}')
		train_data.append(batch[b'data'])
		train_labels.extend(batch[b'labels'])

	train_data = np.concatenate(train_data, axis=0)
	train_labels = np.array(train_labels)

	# Load test data
	test_batch = unpickle(f'{data_dir}/test_batch')
	test_data = test_batch[b'data']
	test_labels = np.array(test_batch[b'labels'])

	# Reshape to (N, C, H, W) and normalize to [0, 1]
	train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
	test_data = test_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

	print(f"Train data range: [{train_data.min()}, {train_data.max()}]")
	print(f"Train data dtype: {train_data.dtype}, shape: {train_data.shape}")

	return (
		torch.from_numpy(train_data),
		torch.from_numpy(train_labels.astype(np.int64)),
		torch.from_numpy(test_data),
		torch.from_numpy(test_labels.astype(np.int64)),
	)


def get_default_device():
	if torch.cuda.is_available():
		return torch.device('cuda')
	
	if torch.backends.mps.is_available():
		return torch.device('mps')
	return torch.device('cpu')


def to_device(data, device):
	if isinstance(data, (list, tuple)):
		return [to_device(x, device) for x in data]
	return data.to(device, non_blocking=True)


class DeviceDataLoader:
	def __init__(self, dl, device):
		self.dl = dl
		self.device = device

	def __iter__(self):
		for b in self.dl:
			yield to_device(b, self.device)

	def __len__(self):
		return len(self.dl)


def accuracy(outputs, labels):
	_, preds = torch.max(outputs, dim=1)
	return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Cifar10CnnModel(nn.Module):
	def __init__(self, dropout=0.3):
		super().__init__()
		self.network = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Dropout2d(dropout),

			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Dropout2d(dropout),

			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Dropout2d(dropout),

			nn.Flatten(),
			nn.Linear(256 * 4 * 4, 1024),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(512, 10),
		)

	def features(self, xb):
		# Everything except the last Linear layer
		return self.network[:-1](xb)

	def forward(self, xb):
		return self.network(xb)


@torch.no_grad()
def evaluate(model, val_loader):
	model.eval()
	outputs = []
	for batch in val_loader:
		images, labels = batch
		out = model(images)
		loss = F.cross_entropy(out, labels)
		acc = accuracy(out, labels)
		outputs.append({'val_loss': loss.detach(), 'val_acc': acc})
	batch_losses = [x['val_loss'] for x in outputs]
	batch_accs = [x['val_acc'] for x in outputs]
	epoch_loss = torch.stack(batch_losses).mean()
	epoch_acc = torch.stack(batch_accs).mean()
	return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


# TODO: these transforms work on tensors but are designed for PIL images — consider using v2 transforms
train_augment = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
])


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
	history = []
	optimizer = opt_func(model.parameters(), lr, weight_decay=1e-4)
	for epoch in range(epochs):
		model.train()
		train_losses = []
		for batch in train_loader:
			images, labels = batch
			images = train_augment(images)
			out = model(images)
			loss = F.cross_entropy(out, labels)
			train_losses.append(loss)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

		result = evaluate(model, val_loader)
		result['train_loss'] = torch.stack(train_losses).mean().item()
		mlflow.log_metrics({
			"train_loss": result['train_loss'],
			"val_loss": result['val_loss'],
			"val_acc": result['val_acc'],
		}, step=epoch)
		print("Epoch [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
			epoch + 1, epochs, result['train_loss'], result['val_loss'], result['val_acc']))
		history.append(result)
	return history


def main():
	# Hyperparameters
	batch_size = 128
	num_epochs = 30
	lr = 0.001
	val_size = 5000
	dropout = 0.3
	weight_decay = 1e-4

	mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050"))
	mlflow.set_experiment("cifar10-cnn")

	with mlflow.start_run():
		mlflow.log_params({
			"batch_size": batch_size,
			"num_epochs": num_epochs,
			"learning_rate": lr,
			"val_size": val_size,
			"optimizer": "Adam",
			"dropout": dropout,
			"weight_decay": weight_decay,
			"augmentation": "random_crop+horizontal_flip",
		})

		# Load data
		print("Loading CIFAR-10 dataset...")
		train_data, train_labels, test_data, test_labels = load_cifar10()

		# Split into train/val
		full_dataset = TensorDataset(train_data, train_labels)
		train_size = len(full_dataset) - val_size
		torch.manual_seed(42)
		train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
		test_ds = TensorDataset(test_data, test_labels)

		# Create data loaders
		train_dl = DataLoader(train_ds, batch_size, shuffle=True)
		val_dl = DataLoader(val_ds, batch_size * 2)
		test_dl = DataLoader(test_ds, batch_size * 2)

		# Setup device
		device = get_default_device()
		print(f"Using device: {device}")
		mlflow.log_param("device", str(device))
		train_dl = DeviceDataLoader(train_dl, device)
		val_dl = DeviceDataLoader(val_dl, device)
		test_dl = DeviceDataLoader(test_dl, device)

		# Create model
		model = to_device(Cifar10CnnModel(), device)

		# Evaluate before training
		print("Initial validation:", evaluate(model, val_dl))

		# Train
		print(f"\nTraining for {num_epochs} epochs...")
		history = fit(num_epochs, lr, model, train_dl, val_dl)

		# Test
		test_result = evaluate(model, test_dl)
		mlflow.log_metrics({
			"test_loss": test_result['val_loss'],
			"test_acc": test_result['val_acc'],
		})
		print("\nTest set evaluation:", test_result)

		# Save and log model
		torch.save(model.state_dict(), 'cifar10-cnn.pth')
		mlflow.log_artifact('cifar10-cnn.pth')
		print("Model saved to cifar10-cnn.pth")

	return history


if __name__ == '__main__':
	main()
