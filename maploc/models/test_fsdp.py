import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as L
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.demos import Transformer, WikiText2


class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        print('init transformer')
        self.model = Transformer(  # 1B parameters
            vocab_size=vocab_size,
            nlayers=32,
            nhid=4096,
            ninp=1024,
            nhead=64,
        )

    def training_step(self, batch):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)


L.seed_everything(42)

# Data
dataset = WikiText2()
train_dataloader = DataLoader(dataset,batch_size=1)

# Model
model = LanguageModel(vocab_size=dataset.vocab_size)

# Trainer
trainer = L.Trainer(accelerator="cuda", devices=4, strategy=FSDPStrategy(sharding_strategy="FULL_SHARD"))
trainer.fit(model, train_dataloader)
trainer.print(torch.cuda.memory_summary())