import lightning as L
import torch
from hydra.utils import instantiate
from monai import metrics as mm


class Net(L.LightningModule):
    def __init__(self, model, criterion, optimizer, lr, scheduler=None):
        super().__init__()
        self.model = model

        self.get_dice = mm.DiceMetric(include_background=False)
        self.get_iou = mm.MeanIoU(include_background=False)
        self.get_recall = mm.ConfusionMatrixMetric(
            include_background=False, metric_name="sensitivity"
        )
        self.get_precision = mm.ConfusionMatrixMetric(
            include_background=False, metric_name="precision"
        )

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Handle both Hydra config and direct class instantiation
        if hasattr(self.optimizer, '_target_'):
            # Hydra configuration object
            optimizer = instantiate(self.optimizer, self.parameters(), lr=self.lr)
        else:
            # Direct class instantiation
            optimizer = self.optimizer(self.parameters(), lr=self.lr)
            
        if self.scheduler:
            if hasattr(self.scheduler, '_target_'):
                # Hydra configuration object
                scheduler = instantiate(self.scheduler, optimizer=optimizer)
            elif callable(self.scheduler):
                # Function that creates scheduler
                scheduler = self.scheduler(optimizer)
            else:
                # Direct class instantiation
                scheduler = self.scheduler(optimizer, T_max=100)  # Default T_max for CosineAnnealingLR
                
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.model.deep_supervision:
            logits, logits_aux = self(x)

            aux_loss = sum(self.criterion(z, y) for z in logits_aux)
            loss = (self.criterion(logits, y) + aux_loss) / (1 + len(logits_aux))

            self.log("train_loss", loss)
            return loss

        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.model.deep_supervision:
            logits, _ = self(x)
        else:
            logits = self(x)

        loss = self.criterion(logits, y)
        self.log("val_loss", loss)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.get_dice(preds, y)
        self.get_iou(preds, y)
        self.get_recall(preds, y)
        self.get_precision(preds, y)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.model.deep_supervision:
            logits, _ = self(x)
        else:
            logits = self(x)

        loss = self.criterion(logits, y)
        self.log("test_loss", loss)

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.get_dice(preds, y)

    def on_validation_epoch_end(self):
        dice = self.get_dice.aggregate().item()        self.get_iou(preds, y)
        iou = self.get_iou.aggregate().item()
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()

        self.log("val_dice", dice)
        self.log("val_iou", iou)    def on_validation_epoch_end(self):
        self.log("val_recall", recall)gate().item()
        self.log("val_precision", precision)gate().item()
        self.log("val_f1", 2 * (precision * recall) / (precision + recall + 1e-8))gate()[0].item()
gate()[0].item()
        self.get_dice.reset()
        self.get_iou.reset()        self.log("val_dice", dice)
        self.get_recall.reset()ou)
        self.get_precision.reset()", recall)
    ", precision)
    def on_test_epoch_end(self):ecision * recall) / (precision + recall + 1e-8))
        dice = self.get_dice.aggregate().item()
        iou = self.get_iou.aggregate().item()
        recall = self.get_recall.aggregate()[0].item()
        precision = self.get_precision.aggregate()[0].item()

        self.log("test_dice", dice)
        self.log("test_iou", iou)    def on_test_epoch_end(self):
        self.log("test_recall", recall)ate().item()
        self.log("test_precision", precision)ate().item()
        self.log("test_f1", 2 * (precision * recall) / (precision + recall + 1e-8))ate()[0].item()
ate()[0].item()
        self.get_dice.reset()
        self.get_iou.reset()        self.log("test_dice", dice)
        self.get_recall.reset()iou)
        self.get_precision.reset()l", recall)
        self.get_dice.reset()
        self.get_iou.reset()
        self.get_recall.reset()
        self.get_precision.reset()
