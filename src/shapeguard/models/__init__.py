from shapeguard.models.unet import UNet


def build_model(config):
    classes = 3 if config["data"]["task"] == "multiclass" else 2
    return UNet(classes, config["model"]["widths"], config["model"]["architecture"] == "attention_unet")
