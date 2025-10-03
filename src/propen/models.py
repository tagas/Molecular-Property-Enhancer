import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class Encode(nn.Module):
    def __init__(self, vocab_size=22):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x):
        return torch.eye(self.vocab_size, device=x.device)[x]
        # return torch.eye(self.vocab_size)[x]

    def decode(self, x):
        return torch.argmax(x, dim=-1)


class ResNetBlock(nn.Module):
    """ResNet architecure with flatten input."""

    def __init__(self, hidden_dim=512, inp_cond_dim=None):
        super().__init__()
        middle_dim = 2 * hidden_dim
        self.inp_cond_dim = inp_cond_dim

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim, middle_dim)

        self.ln2 = nn.LayerNorm(middle_dim)
        self.mlp2 = nn.Linear(middle_dim, hidden_dim)

        self.act = nn.GELU()

    def forward(self, x, cond=None):
        h = self.mlp1(self.act(self.ln1(x)))
        h = self.mlp2(self.act(self.ln2(h)))

        # residual connection
        h += x
        return h


class Encoder(nn.Module):
    """Encoder with ResNet architecure. Maps input to low-dim latent representation."""

    def __init__(
        self,
        vocab_size=22,
        seq_len=149,
        hidden_dim=512,
        num_blocks=1,
        inp_cond_dim=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        print(f"""
              vocab_size: {vocab_size}
              seq_len: {seq_len}
        """)

        # init embedding
        self.mlp = nn.Linear(self.vocab_size * self.seq_len, hidden_dim)

        # resnet blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResNetBlock(hidden_dim, inp_cond_dim=inp_cond_dim))

    def forward(self, x, cond=None):
        # add positional encoding
        # AA level embedding
        """
        p = positionalencoding1d(d_model=20, length=x.shape[1], device=x.get_device()).unsqueeze(0)
        p = torch.tile(p, dims=(x.shape[0], 1, 1))
        xp = torch.cat((x, p), dim=-1)
        xp = self.activation(self.input_aa(xp))
        xp = self.activation(self.conv(xp.permute(0, 2, 1)))
        xp = torch.mean(xp, dim=2)
        """

        # flatten embedding
        out = self.mlp(x.reshape(x.shape[0], self.vocab_size * self.seq_len))

        # concatenate
        # out = torch.cat((out, xp), dim=1)

        for block in self.blocks:
            out = block(out, cond=cond)
        return out


class Decoder(nn.Module):
    """Decoder with ResNet architecure.
    Maps latent low-dim representation back to full-sized input."""

    def __init__(
        self,
        vocab_size=22,
        seq_len=149,
        enc_hidden_dim=256,
        hidden_dim=256,
        num_blocks=1,
        inp_cond_dim=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        # from bottleneck layer to decoder
        self.mlp = nn.Linear(enc_hidden_dim, hidden_dim)

        # resnet blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResNetBlock(hidden_dim, inp_cond_dim=inp_cond_dim))

        # back to original size
        self.proj = nn.Linear(hidden_dim, self.vocab_size * self.seq_len)

    def forward(self, z, cond=None):
        z = self.mlp(z)
        for block in self.blocks:
            z = block(z, cond=cond)
        logits = self.proj(z)
        logits = logits.reshape(logits.shape[0], self.seq_len, self.vocab_size)

        return logits


class AE_matched(pl.LightningModule):
    def __init__(
        self,
        alphabet,
        seq_len: int,
        lr: float = 1e-4,
        wd: float = 1e-4,
        vocab_size=22,
        use_lr_scheduler=False,
        total_steps=None,
    ):
        super(AE_matched, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.wd = wd
        self.vocab_size = vocab_size
        self.encoder = Encoder(
            vocab_size=self.vocab_size, seq_len=seq_len, hidden_dim=128, num_blocks=3
        )
        self.decoder = Decoder(
            vocab_size=self.vocab_size,
            seq_len=seq_len,
            enc_hidden_dim=128,
            hidden_dim=256,
            num_blocks=3,
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.onehot_enc = Encode(vocab_size=self.vocab_size)
        self.alphabet = alphabet
        self.seq_len = seq_len
        self.total_steps = total_steps
        self.use_lr_scheduler = use_lr_scheduler

    def encode_seqs(self, x):
        x_oh = self.onehot_enc(x)
        x_enc = self.encoder(x_oh, cond=None)
        return x_enc

    def forward(self, x):
        encoded = self.encode_seqs(x)
        decoded = self.decoder(encoded)
        return decoded

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        config = {"optimizer": optimizer}
        if self.use_lr_scheduler:
            if self.total_steps is None:
                raise ValueError(
                    "Cannot use learning rate schedulers without specifying total steps."
                )
            # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #     optimizer, total_steps=self.total_steps, max_lr=self.lr * 10
            # )
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=10, factor=0.3
            )
            config["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            }

        return config

    def training_step(self, batch, batch_idx):
        x_i = batch[0]
        x_c = batch[1]
        reconstructed = self(x_i)
        reconstructed = torch.permute(reconstructed, (0, 2, 1))
        target = x_c
        # loss = self.loss(reconstructed, target[: self.seq_len])
        # loss += self.loss(reconstructed, x_i[: self.seq_len])

        loss = self.loss(reconstructed, target)
        loss += self.loss(reconstructed, x_i)

        # Logging
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_i = batch[0]
        x_c = batch[1]
        reconstructed = self(x_i)
        reconstructed = torch.permute(reconstructed, (0, 2, 1))
        target = x_c
        val_loss = self.loss(reconstructed, target)
        val_loss += self.loss(reconstructed, x_i)
        # Logging
        self.log("val_loss", val_loss)
