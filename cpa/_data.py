from typing import Optional

from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter, AnnDataLoader
from scvi.model._utils import parse_device_args


class AnnDataSplitter(DataSplitter):
    def __init__(
            self,
            adata_manager: AnnDataManager,
            train_indices,
            valid_indices,
            test_indices,
            use_gpu: bool = False,
            **kwargs,
    ):
        super().__init__(adata_manager)
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.train_idx = train_indices
        self.val_idx = valid_indices
        self.test_idx = test_indices
        self.device = None
        self.pin_memory = False

    def setup(self, stage: Optional[str] = None):
        # Convert use_gpu to accelerator string
        accelerator = "cuda" if self.use_gpu else "cpu"
        _, _, self.device = parse_device_args(
            accelerator=accelerator, return_device="torch"
        )
        self.pin_memory = (
            True
            if (self.device != "cpu" and accelerator == "cuda")
            else False
        )

    def train_dataloader(self):
        if len(self.train_idx) > 0:
            return AnnDataLoader(
                self.adata_manager,
                indices=self.train_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            data_loader_kwargs = self.data_loader_kwargs.copy()
            # if len(self.valid_indices < 4096):
            #     data_loader_kwargs.update({'batch_size': len(self.valid_indices)})
            # else:
            #     data_loader_kwargs.update({'batch_size': 2048})
            return AnnDataLoader(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return AnnDataLoader(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
