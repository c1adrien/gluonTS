from typing import NamedTuple, Optional
from functools import partial

import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader

from gluonts.env import env
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.model.estimator import Estimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import SelectFields, Transformation
from gluonts.itertools import maybe_len

from pts import Trainer
from pts.model import get_module_forward_input_names
from pts.dataset.loader import TransformedIterableDataset
"""
ici on a construi un pytorchestimator alors qu'il existe in pytorchLightingEstimator disponible sur GluonTS. Les deux dérivent de estimator.  """

class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    predictor: PyTorchPredictor


class PyTorchEstimator(Estimator):
    @validated()
    def __init__(
        self, trainer: Trainer, lead_time: int = 0, dtype: np.dtype = np.float32
      ,**kwargs,
    ) -> None:
        super().__init__(lead_time=lead_time)
        self.trainer = trainer
        self.dtype = dtype
        self.max_idle_transforms = kwargs["max_idle_transforms"] if "max_idle_transforms" in kwargs else None

    def create_transformation(self) -> Transformation:
        """
        Create and return the transformation needed for training and inference.

        Returns
        -------
        Transformation
            The transformation that will be applied entry-wise to datasets,
            at training and inference time.
        """
        raise NotImplementedError

    def create_instance_splitter(self, mode: str) -> Transformation:
        """
        Create and return the instance splitter needed for training, validation or testing.

        Returns
        -------
        Transformation
            The InstanceSplitter that will be applied entry-wise to datasets,
            at training, validation and inference time based on mode.
        """
        raise NotImplementedError

    def create_training_network(self, device: torch.device) -> nn.Module:
        """
        Create and return the network used for training (i.e., computing the
        loss).

        Returns
        -------
        nn.Module
            The network that computes the loss given input data.
        """
        raise NotImplementedError

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: nn.Module,
        device: torch.device,
    ) -> PyTorchPredictor:
        """
        Create and return a predictor object.

        Returns
        -------
        Predictor
            A predictor wrapping a `nn.Module` used for inference.
        """
        raise NotImplementedError

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) -> TrainOutput:
        transformation = self.create_transformation()

        trained_net = self.create_training_network(self.trainer.device)

        input_names = get_module_forward_input_names(trained_net)

        with env._let(max_idle_transforms=self.max_idle_transforms or maybe_len(training_data) or 0):
            training_instance_splitter = self.create_instance_splitter("training")
        training_iter_dataset = TransformedIterableDataset(
            dataset=training_data,
            transform=transformation
            + training_instance_splitter
            + SelectFields(input_names),
            is_train=True,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
        )

        training_data_loader = DataLoader(
            training_iter_dataset,
            batch_size=self.trainer.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
            **kwargs,
        )

        validation_data_loader = None
        if validation_data is not None:
            with env._let(max_idle_transforms=self.max_idle_transforms or maybe_len(validation_data) or 0):
                validation_instance_splitter = self.create_instance_splitter("validation")
            validation_iter_dataset = TransformedIterableDataset(
                dataset=validation_data,
                transform=transformation
                + validation_instance_splitter
                + SelectFields(input_names),
                is_train=True,
                cache_data=cache_data,
            )
            validation_data_loader = DataLoader(
                validation_iter_dataset,
                batch_size=self.trainer.batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True,
                worker_init_fn=self._worker_init_fn,
                **kwargs,
            )

        self.trainer(
            net=trained_net,
            train_iter=training_data_loader,
            validation_iter=validation_data_loader,
        )

        return TrainOutput(
            transformation=transformation,
            trained_net=trained_net,
            predictor=self.create_predictor(
                transformation, trained_net, self.trainer.device
            ),
        )

    @staticmethod
    def _worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) -> PyTorchPredictor:
        return self.train_model(
            training_data,
            validation_data,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            **kwargs,
        ).predictor


#n'est pas utilisé ici. Mais correspond à l'estimateur utilisé dans GluonTS et ses models 

class PyTorchLightningEstimator(Estimator):
    """
    An `Estimator` type with utilities for creating PyTorch-Lightning-based
    models.

    To extend this class, one needs to implement three methods:
    `create_transformation`, `create_training_network`, `create_predictor`,
    `create_training_data_loader`, and `create_validation_data_loader`.
    """

    @validated()
    def __init__(
        self,
        trainer_kwargs: Dict[str, Any],
        lead_time: int = 0,
    ) -> None:
        super().__init__(lead_time=lead_time)
        self.trainer_kwargs = trainer_kwargs

    def create_transformation(self) -> Transformation:
        """
        Create and return the transformation needed for training and inference.

        Returns
        -------
        Transformation
            The transformation that will be applied entry-wise to datasets,
            at training and inference time.
        """
        raise NotImplementedError

    def create_lightning_module(self) -> pl.LightningModule:
        """
        Create and return the network used for training (i.e., computing the
        loss).

        Returns
        -------
        pl.LightningModule
            The network that computes the loss given input data.
        """
        raise NotImplementedError

    def create_predictor(
        self,
        transformation: Transformation,
        module,
    ) -> PyTorchPredictor:
        """
        Create and return a predictor object.

        Parameters
        ----------
        transformation
            Transformation to be applied to data before it goes into the model.
        module
            A trained `pl.LightningModule` object.

        Returns
        -------
        Predictor
            A predictor wrapping a `nn.Module` used for inference.
        """
        raise NotImplementedError

    def create_training_data_loader(
        self, data: Dataset, module, **kwargs
    ) -> Iterable:
        """
        Create a data loader for training purposes.

        Parameters
        ----------
        data
            Dataset from which to create the data loader.
        module
            The `pl.LightningModule` object that will receive the batches from
            the data loader.

        Returns
        -------
        Iterable
            The data loader, i.e. and iterable over batches of data.
        """
        raise NotImplementedError

    def create_validation_data_loader(
        self, data: Dataset, module, **kwargs
    ) -> Iterable:
        """
        Create a data loader for validation purposes.

        Parameters
        ----------
        data
            Dataset from which to create the data loader.
        module
            The `pl.LightningModule` object that will receive the batches from
            the data loader.

        Returns
        -------
        Iterable
            The data loader, i.e. and iterable over batches of data.
        """
        raise NotImplementedError

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        from_predictor: Optional[PyTorchPredictor] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> TrainOutput:
        transformation = self.create_transformation()

        with env._let(max_idle_transforms=max(len(training_data), 100)):
            transformed_training_data: Dataset = transformation.apply(
                training_data, is_train=True
            )
            if cache_data:
                transformed_training_data = Cached(transformed_training_data)

            training_network = self.create_lightning_module()

            training_data_loader = self.create_training_data_loader(
                transformed_training_data,
                training_network,
                shuffle_buffer_length=shuffle_buffer_length,
            )

        validation_data_loader = None

        if validation_data is not None:
            with env._let(max_idle_transforms=max(len(validation_data), 100)):
                transformed_validation_data: Dataset = transformation.apply(
                    validation_data, is_train=True
                )
                if cache_data:
                    transformed_validation_data = Cached(
                        transformed_validation_data
                    )

                validation_data_loader = self.create_validation_data_loader(
                    transformed_validation_data,
                    training_network,
                )

        if from_predictor is not None:
            training_network.load_state_dict(
                from_predictor.network.state_dict()
            )

        monitor = "train_loss" if validation_data is None else "val_loss"
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor=monitor, mode="min", verbose=True
        )

        custom_callbacks = self.trainer_kwargs.pop("callbacks", [])
        trainer = pl.Trainer(
            **{
                "accelerator": "auto",
                "callbacks": [checkpoint] + custom_callbacks,
                **self.trainer_kwargs,
            }
        )

        trainer.fit(
            model=training_network,
            train_dataloaders=training_data_loader,
            val_dataloaders=validation_data_loader,
            ckpt_path=ckpt_path,
        )

        if checkpoint.best_model_path != "":
            logger.info(
                f"Loading best model from {checkpoint.best_model_path}"
            )
            best_model = training_network.__class__.load_from_checkpoint(
                checkpoint.best_model_path
            )
        else:
            best_model = training_network

        return TrainOutput(
            transformation=transformation,
            trained_net=best_model,
            trainer=trainer,
            predictor=self.create_predictor(transformation, best_model),
        )

    @staticmethod
    def _worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> PyTorchPredictor:
        return self.train_model(
            training_data,
            validation_data,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
        ).predictor

    def train_from(
        self,
        predictor: Predictor,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
    ) -> PyTorchPredictor:
        assert isinstance(predictor, PyTorchPredictor)
        return self.train_model(
            training_data,
            validation_data,
            from_predictor=predictor,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
        ).predictor