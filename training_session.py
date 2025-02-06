import os
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Callable, List, Dict, Tuple, Optional, Union, Literal
from torch.utils.data import DataLoader
from tqdm import tqdm
from ModelOps import setup_seed, load_weights, train_model, init_last_biases, get_lr_list
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@dataclass
class TrainingResult:
    train_losses: List[float]
    val_losses: List[float]
    last_save: int
    min_val_loss: float
    lr_history: List[float]
    weights_path: str

class TrainingSession:
    def __init__(
        self,
        model_class: str,
        model_init_args: Optional[Tuple] = (),
        model_init_kwargs: Optional[Dict] = {},
        training_loader: DataLoader = None,
        validation_loader: DataLoader = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        seeds: List[int] = [42],
        weights_path: str = './weights',
        optimizer_class: Optional[Callable] = optim.AdamW,
        optimizer_init_kwargs: Optional[Dict] = {},
        load_weights_from_path: Optional[str] = None,
        init_biases: Optional[List[float]] = None,
        lr: Union[Literal["one_cycle", "cosine", "reduce_on_plateau"], float] = 1e-4,
        weight_decay: float = 0.1,
        session_name: str = "my_session"
    ):
        """
        TrainingSession class to manage the training loop over multiple seeds.
        """
        self.model_class_name = model_class
        self.model_init_args = model_init_args
        self.model_init_kwargs = model_init_kwargs
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.loss_fn = loss_fn
        self.seeds = seeds
        self.weights_path = weights_path
        self.optimizer_class = optimizer_class
        self.optimizer_init_kwargs = optimizer_init_kwargs
        self.load_weights_from_path = load_weights_from_path
        self.init_biases = init_biases
        self.lr = lr
        self.weight_decay = weight_decay
        self.session_name = session_name
        self.training_results: Dict[int, TrainingResult] = {}

    # ------------------- Main methods: train() and plot() -------------------
    def train(
        self,
        epochs: int = 30,
        verbose: bool = True,
        device: str = "cuda",
        unlock_backbone: int = 30
    ) -> Tuple[nn.Module, Dict[int, Dict[str, Union[List[float], int, float, str]]]]:
        """
        Executes the training loop over the specified seeds.
        """
        self.device = device
        self.epochs = epochs
        self.verbose = verbose

        # Map the model_class_name to the actual class
        model_class = self._get_model_class(self.model_class_name)

        for seed in self.seeds:
            setup_seed(seed)
            if self.verbose:
                print(f"Training with seed: {seed}")

            # Initialize model and directories for weights
            model = self._initialize_model(model_class)
            model_weights_dir = os.path.join(self.weights_path, str(seed))
            os.makedirs(model_weights_dir, exist_ok=True)

            # Set optimizer and scheduler
            optimizer, scheduler = self._set_optimizer_and_scheduler(model)

            # Track losses and learning rate history
            train_loss_history, validation_loss_history, lr_history = [], [], []
            min_val_loss, last_saved_epoch = float("inf"), -1

            with tqdm(total=self.epochs, desc="Epochs", disable=False, leave=False) as pbar:
                for epoch in range(self.epochs):
                    if self.verbose:
                        print(f"SEED: {seed} | EPOCH: {epoch}")

                    # Unlock backbone layers if epoch matches unlock_backbone
                    if epoch == unlock_backbone:
                        for param in model.parameters():
                            param.requires_grad = True

                    # Log learning rate
                    if scheduler:
                        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            lr_history.append(optimizer.param_groups[0]["lr"])
                        else:
                            lr_history.append(scheduler.get_last_lr()[0])
                    else:
                        lr_history.append(self.lr if isinstance(self.lr, float) else optimizer.param_groups[0]["lr"])

                    if self.verbose:
                        print("Learning Rate:", lr_history[-1])

                    # Training and validation
                    results = train_model(
                        model=model,
                        dataloader=self.training_loader,
                        val_loader=self.validation_loader,
                        device=self.device,
                        loss_f=self.loss_fn,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        model_class=self.model_class_name.lower(),
                        # use_text_enrich=self.use_text_enrich,
                        # use_pos_tags=self.use_pos_tags,
                        verbose=self.verbose,
                    )
                    train_loss_history.append(results["train_loss"])
                    validation_loss_history.append(results["validation_loss"])

                    # Save best model
                    if validation_loss_history[-1] < min_val_loss:
                        min_val_loss = validation_loss_history[-1]
                        self._save_model(
                            model,
                            os.path.join(model_weights_dir, f"{model.__class__.__name__}_{self.session_name}.pth"),
                            epoch
                        )
                        last_saved_epoch = epoch
                    pbar.update(1)

            # Record training details
            self.training_results[seed] = TrainingResult(
                train_losses=train_loss_history,
                val_losses=validation_loss_history,
                last_save=last_saved_epoch,
                min_val_loss=min_val_loss,
                lr_history=lr_history,
                weights_path=model_weights_dir
            )

    def plot(self, to_folder: Optional[str] = None) -> None:
        """
        Plots the mean (± std) of the training and validation losses across seeds.

        :param to_folder: If specified, saves the plot to the given folder.
        """
        sns.set_style("darkgrid")
        sns.set_palette("pastel")

        # Extract train and val lists from each seed
        train_data = [res.train_losses for res in self.training_results.values()]
        val_data   = [res.val_losses   for res in self.training_results.values()]

        # Ensure consistent epoch lengths (some seeds might have fewer epochs if implemented differently)
        max_epochs = min(len(t) for t in train_data)
        train_data = np.array([t[:max_epochs] for t in train_data])
        val_data   = np.array([v[:max_epochs] for v in val_data])

        # Compute mean and std along the "seeds" axis => shape: (n_epochs,)
        train_mean = train_data.mean(axis=0)
        train_std  = train_data.std(axis=0)
        val_mean   = val_data.mean(axis=0)
        val_std    = val_data.std(axis=0)

        epochs_range = range(1, max_epochs + 1)

        plt.figure(figsize=(8, 6))

        # Plot training loss
        plt.plot(epochs_range, train_mean, label="Train Loss", color='tab:blue')
        plt.fill_between(
            epochs_range,
            train_mean - train_std,
            train_mean + train_std,
            color='tab:blue',
            alpha=0.2
        )

        # Plot validation loss
        plt.plot(epochs_range, val_mean, label="Validation Loss", color='tab:orange')
        plt.fill_between(
            epochs_range,
            val_mean - val_std,
            val_mean + val_std,
            color='tab:orange',
            alpha=0.2
        )

        plt.title(f"{self.model_class_name.capitalize()} - {self.session_name.replace('_', ' ').capitalize()} - Loss across Seeds (Mean ± Std)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.tight_layout()

        # Save if requested
        if to_folder is not None:
            os.makedirs(to_folder, exist_ok=True)
            file_path = os.path.join(to_folder, f"{self.model_class_name}_{self.session_name}_loss.png")
            plt.savefig(file_path, dpi=150)

        plt.show()
    
    # --------------------------- Private Methods ---------------------------

    def _get_model_class(self, model_class_name: str) -> Callable[..., nn.Module]:
        """
        Maps the model class name string to the actual class.

        :param model_class_name: Name of the model class.
        :type model_class_name: str
        :return: The model class.
        :rtype: Callable[..., nn.Module]
        """
        if model_class_name.lower() == 'gru':
            from models import Gru
            return Gru
        elif model_class_name.lower() == 'grubert':
            from models import GruBERT
            return GruBERT
        else:
            raise ValueError(f"Unknown model class name: {model_class_name}")

    def _save_model(self, model: nn.Module, path: str, epoch: int) -> None:
        """
        Save model weights to a specified path.

        :param model: Model whose weights are to be saved.
        :type model: nn.Module
        :param path: Directory path for saving the model.
        :type path: str
        :param epoch: Epoch at which the model is saved.
        :type epoch: int
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

        if self.verbose:
            print(f"\t* Model saved at epoch {epoch} *")

    def _set_optimizer_and_scheduler(
        self,
        model: nn.Module
    ) -> Tuple[optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Sets the optimizer and scheduler based on the learning rate strategy.

        :param model: Model to optimize.
        :type model: nn.Module
        :return: Configured optimizer and scheduler.
        :rtype: Tuple[optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]
        """
        lr = self.lr if isinstance(self.lr, float) else 1e-4
        params = get_lr_list(model, lr_f=lr)
        optimizer = self.optimizer_class(
            params=params,
            lr=lr,
            weight_decay=self.weight_decay,
            **self.optimizer_init_kwargs
        )

        scheduler = None
        if self.lr == "one_cycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=5e-4,
                epochs=self.epochs,
                steps_per_epoch=len(self.training_loader)
            )
        elif self.lr == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=1e-6
            )
        elif self.lr == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=3,
                min_lr=1e-6
            )
        return optimizer, scheduler

    def _initialize_model(self, model_class: Callable[..., nn.Module]) -> nn.Module:
        """
        Initializes the model with pre-trained weights and biases, if specified.

        :param model_class: The class of the model to instantiate.
        :type model_class: Callable[..., nn.Module]
        :return: Initialized model ready for training.
        :rtype: nn.Module
        """
        # Instantiate a new model for each seed
        model = model_class(*self.model_init_args, **self.model_init_kwargs)
        if self.load_weights_from_path:
            load_weights(model, path=self.load_weights_from_path)
        if self.init_biases:
            init_last_biases(model, torch.tensor(self.init_biases, dtype=torch.float32).to(self.device))
        model = model.to(self.device)
        return model
    
