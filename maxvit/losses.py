import torch


def fc_tversky(y_true, y_pred, beta, gamma=0.75, batch_mean=True):
    smooth = 1
    tp = torch.sum(y_true * y_pred, axis=1)
    fn = torch.sum((1. - y_pred) * y_true, axis=1)
    fp = torch.sum(y_pred * (1. - y_true), axis=1)
    tversky = (tp + smooth) /\
        (tp + beta * fn + (1. - beta) * fp + smooth)

    focal_tversky_loss = 1 - tversky

    if batch_mean:
        return torch.pow(torch.mean(focal_tversky_loss), gamma)
    else:
        return torch.pow(focal_tversky_loss, gamma)


classification_loss = torch.nn.MSELoss()
