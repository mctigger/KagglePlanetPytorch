from torch.nn import BCELoss


def PseudoLabelingLoss(loss_fn=BCELoss()):
    def call(prediction, y, i, epoch):
        i = i.data
        i_supervised = (i <= 0).nonzero().squeeze()
        i_unsupervised = (i > 0).nonzero().squeeze()

        supervised_prediction = prediction[i_supervised]
        supervised_y = y[i_supervised]
        unsupervised_prediction = prediction[i_unsupervised]
        unsupervised_y = y[i_unsupervised]

        supervised_loss = loss_fn(supervised_prediction, supervised_y)
        unsupervised_loss = loss_fn(unsupervised_prediction, unsupervised_y)

        a = max((epoch-12), 0) / 35

        loss = (supervised_loss + a * unsupervised_loss) / (1 + a)

        return loss

    return call