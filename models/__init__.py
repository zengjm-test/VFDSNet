from models.vfds import VFDSNet

from models.loss import model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test

__models__ = {
    "vfdsnet": VFDSNet,

}
