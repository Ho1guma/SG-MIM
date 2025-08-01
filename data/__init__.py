from .data_sgmim import build_loader_sgmim
from .data_finetune import build_loader_finetune

def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        return build_loader_sgmim(config, logger)
    else:
        return build_loader_finetune(config, logger)