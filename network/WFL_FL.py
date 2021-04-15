
import torch
import torch.nn.functional as F
import math

def errorloss(predictions, gt):
    # make sure preictions after softmax
    # https://blog.csdn.net/guofei_fly/article/details/104486708
    # clone() 与源张量不共享数据内存，但提供梯度的回溯
    # detach() 与源张量共享数据内存，但不提供梯度计算
    # clone().detach() 既不数据共享，也不对梯度共享

    """
    Shape:
        predictions: [N, C, H, W]
        gt: [N, H, W]
    """

    # predictions = F.softmax(predictions, 1)

    pred = F.softmax(predictions, 1).clone()

    # rank map
    assert pred.size(1) != 1
    sort_pred, _ = torch.sort(pred, dim=1, descending=True, out=None)
    rank_error_map = torch.sub(sort_pred[:, 0, ...], sort_pred[:, 1, ...])

    # error map
    # _, pred = torch.max(pred, 1)
    # error = pred ^ gt
    # error_map = torch.where(error == 0, 0, 1)

    # combime
    # assert error_map.shape == rank_error_map.shape
    # weight_map = torch.add(rank_error_map, error_map)
    weight_map = 1 - rank_error_map
    # weight_map = torch.exp(weight_map)
    # print("wm:",weight_map.shape)

    pred1 = torch.log_softmax(predictions,dim=1)
    # print("pr:",pred1.shape)
    loss_errormap = F.nll_loss(weight_map.unsqueeze(1)*pred1, gt ,weight=None, ignore_index=255,reduction='mean')
    loss_focal = F.nll_loss((torch.tensor(1)-torch.exp(pred1))**2*pred1,gt,weight=None,ignore_index=255,reduction='mean')
    loss = loss_errormap + 0.25*loss_focal
    # loss = 0.25 * loss_focal

    #
    # log_P = torch.log(predictions+(1e-10))
    # one_hot = torch.zeros(predictions.shape, dtype=torch.float).scatter_(1, torch.unsqueeze(gt, dim=1), 1)
    #
    # loss_errormap = -torch.mul(weight_map.unsqueeze(1), one_hot * log_P)
    # loss_focal = -torch.mul((torch.tensor(1)-torch.exp(one_hot*log_P))**2, one_hot * log_P)
    # loss = loss_errormap + 0.25*loss_focal
    # loss = loss_focal.mean()

    return loss


if __name__ == "__main__":
    pred = torch.randn(2, 6, 64, 64)
    gt = torch.randint(0, 6, (2, 64, 64))
    value = errorloss(pred, gt)
    print(value)