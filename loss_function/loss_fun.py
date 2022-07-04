import torch


def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def multilabel_categorical_crossentropy_labelsmooth(y_true, y_pred):
    """多标签分类的交叉熵
        说明：
            1. y_true和y_pred的shape一致，y_true的元素是0～1
               的数，表示当前类是目标类的概率；
            2. 请保证y_pred的值域是全体实数，换言之一般情况下
               y_pred不用加激活函数，尤其是不能加sigmoid或者
               softmax；
            3. 预测阶段则输出y_pred大于0的类；
            4. 详情请看：https://kexue.fm/archives/7359 和
               https://kexue.fm/archives/9064 。
        """
    y_true = y_true.float()  # 先变为float
    infinity = 1e+12
    epsilon = 0.1
    y_mask = y_pred > -infinity / 10
    n_mask = (y_true < 1 - epsilon) & y_mask
    p_mask = (y_true > epsilon) & y_mask
    y_true = torch.clip(y_true, epsilon, 1 - epsilon)
    infs = torch.zeros_like(y_pred) + infinity
    y_neg = torch.where(n_mask, y_pred, -infs) + torch.log(1 - y_true)
    y_pos = torch.where(p_mask, -y_pred, -infs) + torch.log(y_true)
    zeros = torch.zeros_like(y_pred[..., :1])
    y_neg = torch.cat([y_neg, zeros], dim=-1)
    y_pos = torch.cat([y_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pos, dim=-1)
    return neg_loss + pos_loss


def global_pointer_crossentropy(y_true, y_pred, alpha=4):
    """给GlobalPointer设计的交叉熵
    """
    bh = y_pred.shape[0] * y_pred.shape[1]
    y_pred = torch.reshape(y_pred, (bh, -1))
    y_true = torch.reshape(y_true, (bh, -1))
    return torch.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_crossentropy_labelsmooth(y_true, y_pred, alpha=4):
    """给GlobalPointer设计的交叉熵
    """
    bh = y_pred.shape[0] * y_pred.shape[1]
    y_pred = torch.reshape(y_pred, (bh, -1))
    y_true = torch.reshape(y_true, (bh, -1))
    return torch.mean(multilabel_categorical_crossentropy_labelsmooth(y_true, y_pred))
