from shapeguard.losses.boundary import boundary_loss
from shapeguard.losses.geometry import geometry_loss
from shapeguard.losses.ici import ici_terms
from shapeguard.losses.segmentation import global_loss


def objective(logits, batch, config):
    weights = config["loss"]
    target, valid = batch["mask"], batch["valid"]
    fracture_class = logits.shape[1] - 1
    probability = logits.softmax(1)[:, fracture_class]
    fracture = target == fracture_class
    ce, dice = global_loss(logits, target, valid, weights["class_weights"])
    zero = logits.sum() * 0
    instance, center = zero, zero
    if weights["instance"] or weights["center"]:
        instance, center = ici_terms(
            probability, fracture, valid, weights["threshold"], weights["center_size"], bool(weights["center"])
        )
    shape = geometry_loss(probability, fracture, batch["instances"], valid, weights["context"], weights["geometry_mode"]) if weights["geometry"] else zero
    edge = boundary_loss(probability, fracture, batch["instances"], valid, weights) if weights["boundary"] else zero
    values = {"ce": ce, "dice": dice, "instance": instance, "center": center, "geometry": shape, "boundary": edge}
    loss = sum(weights[key] * value for key, value in values.items())
    return loss, {key: float(value.detach()) for key, value in values.items()}
