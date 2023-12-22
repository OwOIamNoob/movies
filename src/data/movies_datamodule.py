from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose
from torchvision import transforms

# comment when do training
import pyrootutils
pyrootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

from src.data.components.vocab import Vocab
from src.data.components.ml_dataset import MLDataset, MLTransformedDataset, Collator


class MLDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str, 
        img_dir:str,
        genre_path: str,
        word_path:str,
        max_seq_len:int,
        rating_transforms: Optional[Compose] = Compose([transforms.ToTensor()]),
        transforms: Optional[Compose] = Compose([transforms.ToTensor()]),
        test_transforms: Optional[Compose] = Compose([transforms.ToTensor()]),
        img_size=256,
        rating_img_size = 64, 
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        masked_language_model: bool = False
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.transforms = transforms
        self.test_transforms = test_transforms
        self.rating_transforms = rating_transforms
        self.prepare_data()
    
    def prepare_data(self) -> None:
        self.title_vocab = Vocab(self.hparams.word_path)
        self.genre_vocab = Vocab(self.hparams.genre_path)
        
        
    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    
        if not self.data_train and not self.data_val and not self.data_test:
            self.train_dataset = MLDataset(
                data_dir=self.hparams.data_dir,
                img_dir =self.hparams.img_dir,
                title_vocab=self.title_vocab,
                genre_vocab=self.genre_vocab,
                data_type="train"
            )
            
            self.val_dataset = MLDataset(
                data_dir=self.hparams.data_dir,
                img_dir =self.hparams.img_dir,
                title_vocab=self.title_vocab,
                genre_vocab=self.genre_vocab,
                data_type="test"
            )
            self.test_dataset = MLDataset(
                data_dir=self.hparams.data_dir,
                img_dir =self.hparams.img_dir,
                title_vocab=self.title_vocab,
                genre_vocab=self.genre_vocab,
                data_type="test"
            )
            
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        self.train_transformed_dataset = MLTransformedDataset(
            dataset = self.train_dataset,
            pad_id = self.title_vocab.vocab['<PAD>'],
            transforms = self.transforms,
            rating_transforms = self.rating_transforms,
            rating_img_size = self.hparams.rating_img_size
        )
        
        return DataLoader(
            dataset = self.train_transformed_dataset,
            batch_size = self.batch_size_per_device,
            shuffle = True,
            num_workers = self.hparams.num_workers,
            collate_fn = Collator(max_seq_len=self.hparams.max_seq_len,
                                  taget_vocab=self.genre_vocab,
                                  pad_id=self.title_vocab.vocab['<PAD>'],
                                  rating_img_size=self.hparams.rating_img_size,
                                  img_size=self.hparams.img_size),
            pin_memory = self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        self.val_transformed_dataset = MLTransformedDataset(
            dataset = self.val_dataset,
            pad_id = self.title_vocab.vocab['<PAD>'],
            transforms = self.test_transforms,
            rating_transforms = self.rating_transforms,
            rating_img_size = self.hparams.rating_img_size
        )
        
        return DataLoader(
            dataset = self.val_transformed_dataset,
            batch_size = self.batch_size_per_device,
            shuffle = True,
            num_workers = self.hparams.num_workers,
            collate_fn = Collator(max_seq_len=self.hparams.max_seq_len,
                                  taget_vocab=self.genre_vocab,
                                  pad_id=self.title_vocab.vocab['<PAD>'],
                                  rating_img_size=self.hparams.rating_img_size,
                                  img_size=self.hparams.img_size),
            pin_memory = self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        self.test_transformed_dataset = MLTransformedDataset(
            dataset = self.test_dataset,
            pad_id = self.title_vocab.vocab['<PAD>'],
            transforms = self.test_transforms,
            rating_transforms = self.rating_transforms,
            rating_img_size = self.hparams.rating_img_size
        )
        
        return DataLoader(
            dataset = self.test_transformed_dataset,
            batch_size = self.batch_size_per_device,
            shuffle = False,
            num_workers = self.hparams.num_workers,
            collate_fn = Collator(max_seq_len=self.hparams.max_seq_len,
                                  target_vocab=self.genre_vocab,
                                  pad_id=self.title_vocab.vocab['<PAD>'],
                                  rating_img_size=self.hparams.rating_img_size,
                                  img_size=self.hparams.img_size),
            pin_memory = self.hparams.pin_memory,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

############################################################### TEST ###############################################################

if __name__ == "__main__":
    datamodule: LightningDataModule = MLDataModule(data_dir="/work/hpc/potato/movies/data/movies/dataset",
                                                   img_dir="/work/hpc/potato/movies/data/movies/dataset/ml1m-images/",
                                                   genre_path="/work/hpc/potato/movies/data/movies/dataset/genres.txt",
                                                   word_path="/work/hpc/potato/movies/data/movies/dataset/words.txt",
                                                   max_seq_len=10,
                                                   rating_transforms=Compose([torch.FloatTensor,
                                                                              transforms.Normalize(mean=[0.5, 3, 10, 2.5],
                                                                                                   std=[0.5, 3, 10, 2.5])]),
                                                   transforms=Compose([ transforms.RandomAffine(degrees=(-10, 10),
                                                                                                translate=(0.1, 0.1),
                                                                                                interpolation=transforms.InterpolationMode.NEAREST),
                                                                        transforms.Resize((256, 256)),
                                                                        transforms.ToTensor()]),
                                                   test_transforms=Compose([transforms.Resize((256, 256)), 
                                                                            transforms.ToTensor()]),
                                                   rating_img_size= 64,
                                                   img_size=256,
                                                   batch_size=8,
                                                   num_workers=4,
                                                   pin_memory=False)
    datamodule.setup()
    dataloader = datamodule.test_dataloader()
    batch = next(iter(dataloader))
    print(batch['titles'])
    print(datamodule.title_vocab.vocab['<PAD>'])