"""
Deep Learning Module for House Price Prediction
PyTorch Neural Network implementation - Bonus Section
Consistent with notebooks/02_Training.ipynb
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import joblib
from pathlib import Path

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Install with: pip install torch")

from .config import MODELS_DIR, RANDOM_STATE


def check_pytorch_available() -> bool:
    """Check if PyTorch is installed."""
    return PYTORCH_AVAILABLE


def get_device() -> 'torch.device':
    """Get the best available device (GPU/MPS/CPU)."""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed")
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


class HousePriceNN(nn.Module):
    """
    Neural Network for House Price Prediction.
    Architecture consistent with notebooks/02_Training.ipynb
    
    Architecture:
    - Input -> 256 -> BatchNorm -> ReLU -> Dropout(0.3)
    - 256 -> 128 -> BatchNorm -> ReLU -> Dropout(0.2)
    - 128 -> 64 -> BatchNorm -> ReLU -> Dropout(0.1)
    - 64 -> 32 -> ReLU
    - 32 -> 1 (Output)
    """
    
    def __init__(self, input_dim: int):
        super(HousePriceNN, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 4
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Output
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class NeuralNetworkTrainer:
    """Trainer class for PyTorch Neural Network."""
    
    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        random_state: int = RANDOM_STATE
    ):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Get device
        self.device = get_device()
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize model
        self.model = HousePriceNN(input_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def _prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> DataLoader:
        """Prepare DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        train_loader = self._prepare_data(X_train, y_train)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
        
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()
                
                self.history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions.flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Returns:
            Dictionary with metrics (MSE, RMSE, R2)
        """
        from sklearn.metrics import mean_squared_error, r2_score
        
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
    
    def save(self, filepath: str = None):
        """Save model to disk."""
        if filepath is None:
            filepath = MODELS_DIR / 'neural_network.pt'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, filepath)
        print(f"✅ Neural network saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = None) -> 'NeuralNetworkTrainer':
        """Load model from disk."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load the model")
        
        if filepath is None:
            filepath = MODELS_DIR / 'neural_network.pt'
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        trainer = cls(input_dim=checkpoint['input_dim'])
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.history = checkpoint['history']
        trainer.best_val_loss = checkpoint['best_val_loss']
        
        return trainer


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[NeuralNetworkTrainer, Dict]:
    """
    Convenience function to train a neural network.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Tuple of (trainer, metrics)
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    input_dim = X_train.shape[1]
    
    trainer = NeuralNetworkTrainer(
        input_dim=input_dim,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )
    
    print(f"\n{'='*60}")
    print("🧠 Training Neural Network")
    print(f"{'='*60}")
    print(f"Input dimension: {input_dim}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val) if X_val is not None else 'N/A'}")
    
    trainer.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    if X_val is not None:
        metrics = trainer.evaluate(X_val, y_val)
        print(f"\n📊 Validation Results:")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   R²: {metrics['R2']:.4f}")
    else:
        metrics = trainer.evaluate(X_train, y_train)
    
    return trainer, metrics


if __name__ == "__main__":
    # Test the module
    if PYTORCH_AVAILABLE:
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"🖥️  Device: {get_device()}")
        
        # Test with random data
        X = np.random.randn(100, 20).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        trainer, metrics = train_neural_network(
            X[:80], y[:80], X[80:], y[80:], epochs=10
        )
        print(f"Test complete! R² = {metrics['R2']:.4f}")
    else:
        print("❌ PyTorch not installed")

