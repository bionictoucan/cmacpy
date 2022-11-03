import torch
from typing import Optional, Callable, Union, Dict, Tuple
from tqdm import tqdm
import os


class Trainer:
    """
    This is the default class for defining trainers to teach neural networks
    different tasks.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    optimiser : torch.optim.Optimizer
        The optimiser to use during training.
    loss_fn : Callable
        The loss function to use during training.
    no_of_epochs : int
        The number of epochs to train for.
    batch_size : int
        The batch size to use.
    data_pth : str
        The path to the training/validation data.
    save_dir : str, optional
        The directory to save the trained models to. Default is ``"./"`` -- the
        :abbr:`CWD (current working directory)`.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        The way to adaptively change the learning rate while training. Default
        is None. For examples on how to adaptively change the learning rate
        please see
        `here <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
    device_id : int or str, optional
        Which device to perform training on. Providing an integer will point to
        the GPU with that specific ID whereas the string option can be used to
        also specify training on the CPU. When using ``nn.DataParallel`` for the
        model, setting ``device_id = "cuda"`` uses all GPUs. Default is 0 -- use
        the GPU corresponding to ID ``"cuda:0"``.
    data_parallel : bool, optional
        Whether or not to use multiple GPUs for training. Default is ``False``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        loss_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
        no_of_epochs: int,
        batch_size: int,
        data_pth: str,
        save_dir: str = "./",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device_id: Union[int, str] = 0,
        data_parallel: bool = False,
    ) -> None:

        self.device = torch.device(
            f"cuda:{device_id}" if type(device_id) == int else device_id
        )

        if data_parallel:
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model
        self.model.to(self.device)

        self.optimiser = optimiser

        self.loss_fn = loss_fn

        self.total_epochs = no_of_epochs
        self.batch_size = batch_size

        self.data_pth = data_pth

        self.save_dir = save_dir

        self.scheduler = scheduler

        self.current_epoch = 0

    def load_data(self) -> None:
        """
        User-defined data loading instance method.
        This should populate self.train_in, self.train_out, self.val_in and self.val_out.
        """

        raise NotImplementedError("This must be user-defined!!")

    def checkpoint(self, add_info: Optional[Dict] = None) -> None:
        """
        This class method creates a checkpoint for the current epoch.

        Parameters
        ----------
        add_info : dict, optional
            An additional information to add to the checkpoint e.g. scheduler
            state dictionary. Default is ``None``.
        """

        if isinstance(self.model, torch.nn.DataParallel):
            self.chkpt = {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }
        else:
            self.chkpt = {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }

        if isinstance(add_info, dict):
            self.chkpt.update(add_info)

    def save_checkpoint(self, custom_path: Optional[str] = None) -> None:
        """
        This class method saves the current checkpoint to the save directory
        defined when instantiating the class.

        Parameters
        ----------
        custom_path : str, optional
            Path to save the model to. Default is ``None``.
        """

        if isinstance(custom_path, str):
            save_pth = f"{custom_path}{self.current_epoch}.pth"
        else:
            save_pth = f"{self.save_dir}{self.current_epoch}.pth"

        if not hasattr(self, "chkpt"):
            self.checkpoint()
        torch.save(self.chkpt, save_pth)

    def load_checkpoint(self, filename: str) -> None:
        """
        This class method loads a checkpoint for the model.

        Parameters
        ----------
        filename : str
            The path to the model to load.
        """

        if os.path.isfile(filename):
            print(f"=> loading checkpoint at {filename}")
            chkpt = torch.load(filename)
            self.current_epoch = chkpt["epoch"]
            self.train_losses = chkpt["train_losses"]
            self.val_losses = chkpt["val_losses"]
            self.model.load_state_dict(chkpt["model_state_dict"])
            self.optimiser.load_state_dict(chkpt["optimiser_state_dict"])

            def_keys = [
                "epoch",
                "train_losses",
                "val_losses",
                "model_state_dict",
                "optimiser_state_dict",
            ]
            if len(chkpt.keys()) != 5:
                self.add_info = {}
                for key in chkpt.keys():
                    if key not in def_keys:
                        self.add_info.update({key: chkpt[key]})
            if self.scheduler:
                self.scheduler.load_state_dict(self.add_info["scheduler_state_dict"])

            train_l = self.train_losses[-1]
            val_l = self.val_losses[-1]
            print(
                f"=> loaded checkpoint at {filename} at epoch {self.current_epoch} with training loss {train_l} and validation loss {val_l}."
            )
        else:
            print(f"=> no checkpoint found at {filename}")


class ClassifierTrainer(Trainer):
    """
    A trainer for training a neural network for classification.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    optimiser : torch.optim.Optimizer
        The optimiser to use during training.
    loss_fn : Callable
        The loss function to use during training.
    no_of_epochs : int
        The number of epochs to train for.
    batch_size : int
        The batch size to use.
    data_pth : str
        The path to the training/validation data.
    save_dir : str, optional
        The directory to save the trained models to. Default is ``"./"`` -- the
        :abbr:`CWD (current working directory)`.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        The way to adaptively change the learning rate while training. Default
        is None. For examples on how to adaptively change the learning rate
        please see
        `here <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
    device_id : int or str, optional
        Which device to perform training on. Providing an integer will point to
        the GPU with that specific ID whereas the string option can be used to
        also specify training on the CPU. When using ``nn.DataParallel`` for the
        model, setting ``device_id = "cuda"`` uses all GPUs. Default is 0 -- use
        the GPU corresponding to ID ``"cuda:0"``.
    data_parallel : bool, optional
        Whether or not to use multiple GPUs for training. Default is ``False``.
    """

    def train(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        The function used to train the classifier model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The data to train the model on wrapped up in a `PyTorch DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the training data.
        """
        self.model.train()

        batch_losses = []
        total, correct = 0.0, 0.0
        for j, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.float().to(self.device), labels.long().to(
                self.device
            )

            self.optimiser.zero_grad()
            output = self.model(images)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimiser.step()

            batch_losses.append(loss.item())

            # work out the % correct per epoch accumulatively
            _, predicted = torch.max(
                output.data, 1
            )  # output will have dimensions (batch_size, output_features) with the maximum taken along the output_features axis i.e. the probability distribution of classes, predicted is populated by the indices of the maximum in this dimension -- the class label.
            total += labels.size(0)
            correct += (
                predicted == labels
            ).sum()  # produces a boolean tensor which when summed evaluates True to 1 and False to 0.

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)

    def validation(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        The function used to validate the classifier model.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            The data to validate the model on wrapped up in a `PyTorch
            DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the validation data.
        """
        self.model.eval()

        total, correct = 0.0, 0.0
        with torch.no_grad():
            batch_losses = []
            for images, labels in val_loader:
                images, labels = images.float().to(self.device), labels.long().to(
                    self.device
                )
                output = self.model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

                loss = self.loss_fn(output, labels)
                batch_losses.append(loss.item())

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)


class RegressorTrainer(Trainer):
    """
    A trainer for training a neural network for regression.


    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    optimiser : torch.optim.Optimizer
        The optimiser to use during training.
    loss_fn : Callable
        The loss function to use during training.
    no_of_epochs : int
        The number of epochs to train for.
    batch_size : int
        The batch size to use.
    data_pth : str
        The path to the training/validation data.
    save_dir : str, optional
        The directory to save the trained models to. Default is ``"./"`` -- the
        :abbr:`CWD (current working directory)`.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        The way to adaptively change the learning rate while training. Default
        is None. For examples on how to adaptively change the learning rate
        please see
        `here <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
    device_id : int or str, optional
        Which device to perform training on. Providing an integer will point to
        the GPU with that specific ID whereas the string option can be used to
        also specify training on the CPU. When using ``nn.DataParallel`` for the
        model, setting ``device_id = "cuda"`` uses all GPUs. Default is 0 -- use
        the GPU corresponding to ID ``"cuda:0"``.
    data_parallel : bool, optional
        Whether or not to use multiple GPUs for training. Default is ``False``.
    """

    def train(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        The function used to train the regression model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The data to train the model on wrapped up in a `PyTorch DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
        """
        self.model.train()

        batch_losses = []
        for j, (inputs, outputs) in enumerate(tqdm(train_loader)):
            inputs, outputs = inputs.float().to(self.device), outputs.float().to(
                self.device
            )

            self.optimiser.zero_grad()
            model_outputs = self.model(inputs)
            loss = self.loss_fn(model_outputs, outputs)
            loss.backward()
            self.optimiser.step()

            batch_losses.append(loss.item())

        return torch.mean(torch.tensor(batch_losses))

    def validation(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        The function used to validate the regression model.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            The data to validate the model on wrapped up in a `PyTorch
            DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
        """
        self.model.eval()

        batch_losses = []
        with torch.no_grad():
            for inputs, outputs in val_loader:
                inputs, outputs = inputs.float().to(self.device), outputs.float().to(
                    self.device
                )
                model_outputs = self.model(inputs)
                loss = self.loss_fn(model_outputs, outputs)

                batch_losses.append(loss.item())

        return torch.mean(torch.tensor(batch_losses))
