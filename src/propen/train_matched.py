import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from .constants import *
from .models import AE_matched
from .utils import Alphabet, DatasetPPI_matched, PaddCollator, read_toks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_propen(
    df_paired: pd.DataFrame,
    output_dir: str,
    lr=1e-4,
    weight_decay=1e-4,
    epochs=300,
    batch_size=128,
    target="HER2",
    model_name="propen",
    seed=1,
    use_lr_scheduler=False,
    num_workers=0,
    heavy_light=1,
    tokenizer="protein_seq_toks",
) -> str:
    # wandb set-up ##########################################
    wandb.login(host="https://genentech.wandb.io/")
    wandb.init(
        project="propen_models_test",
        # id=target + "_" + model_name,
        name=target + "_" + model_name,
    )
    wandb_logger = WandbLogger(
        project="propen_models_test",
        # id=target + "_" + model_name,
        name=target + "_" + model_name,
    )

    # load data ############################################
    print(df_paired.shape)

    print(df_paired.columns)

    s1_max_len = df_paired["s1"].apply(len).max()
    s2_max_len = df_paired["s2"].apply(len).max()
    seq_len_adaptive = max(s1_max_len, s2_max_len)

    if heavy_light:
        print("Training on both heavy and light")
        seq_len = 298
    else:
        print("Training on only heavy chain")
        seq_len = seq_len_adaptive

    print("seq_len", seq_len)

    if tokenizer == "custom_seq_toks":
        custom_seq_toks = read_toks()
        print(custom_seq_toks["toks"])
        alphabet = Alphabet(
            standard_toks=custom_seq_toks["toks"], prepend_bos=False, append_eos=False
        )
        vocab_size = len(custom_seq_toks["toks"]) + 1
    else:
        alphabet = Alphabet(
            standard_toks=eval(tokenizer)["toks"], prepend_bos=False, append_eos=False
        )
        vocab_size = len(eval(tokenizer)["toks"]) + 1
    print(vocab_size)

    # split data into train/valid
    test_heavy_seq = df_paired.iloc[0]["s1"]
    print(test_heavy_seq)
    # print(test_heavy_seq.shape)
    df_paired = df_paired[df_paired.s1 != test_heavy_seq].reset_index(drop=True)
    df_valid = df_paired.sample(frac=0.1)
    df_train = df_paired.drop(df_valid.index)
    print("Train/Validation {}/{}".format(df_train.shape, df_valid.shape))

    # loaders ################################################
    collate_padd = PaddCollator(alphabet, max_len=seq_len)

    train_loader = DataLoader(
        DatasetPPI_matched(df_train, col_names=["s1", "s2"]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_padd,
        num_workers=num_workers,
    )

    valid_loader = DataLoader(
        DatasetPPI_matched(df_valid, col_names=["s1", "s2"]),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_padd,
        num_workers=num_workers,
    )

    # PyTorch Lightning trainer #############################
    model = AE_matched(
        lr=lr,
        vocab_size=vocab_size,
        alphabet=alphabet,
        seq_len=seq_len,
        wd=weight_decay,
        use_lr_scheduler=use_lr_scheduler,
        total_steps=len(df_train) // batch_size * epochs,
    )
    model.to(device)
    model.train()
    print(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/best_model", save_top_k=2, monitor="val_loss", save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    print("Training started...")
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=1,
        enable_progress_bar=True,
        callbacks=callbacks,
    )

    # train #################################################
    print("# Training commences ##########")
    trainer.fit(model, train_loader, valid_loader)
    print("# Training done ###############")
    return checkpoint_callback.best_model_path
