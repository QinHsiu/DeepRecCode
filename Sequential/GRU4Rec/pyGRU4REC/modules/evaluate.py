import torch
import math

def get_recall(indices, targets):
    """ Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
    """
    # targets = targets.view(-1, 1).expand_as(indices)  # (Bxk)
    # #print(targets)
    # hits = (targets == indices).nonzero()
    # print(hits)
    # if len(hits) == 0: return 0
    # n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    # print(n_hits)
    # recall = n_hits / targets.size(0)
    indices=indices.cpu().numpy().tolist()
    targets=targets.cpu().numpy().tolist()
    hints=0
    for user_id in range(len(targets)):
        if targets[user_id] in indices[user_id]:
            hints+=1
    return hints/len(targets)







def get_mrr(indices, targets):
    """ Calculates the MRR score for the given predictions and targets
    
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        mrr (float): the mrr score
    """
    targets = targets.view(-1,1).expand_as(indices)
    # ranks of the targets, if it appears in your indices
    hits = (targets == indices).nonzero()
    if len(hits) == 0: return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(rranks).data / targets.size(0)
    mrr = mrr.item()
    
    return mrr



def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = 1
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j]==actual[user_id]) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def evaluate(logits, targets, k=20):
    """ Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    if isinstance(k,list):
        recalls=[list() for _ in range(len(k))]
        ndcgs=[list() for _ in range(len(k))]
        mrrs=[list() for _ in range(len(k))]
        for i,_k in enumerate(k):
            _, indices = torch.topk(logits, _k, -1)
            recall = get_recall(indices, targets)
            mrr = get_mrr(indices, targets)
            ndcg=ndcg_k(targets,indices,_k)
            recalls[i].append(recall)
            ndcgs[i].append(ndcg)
            mrrs[i].append(mrr)

    else:
        _, indices = torch.topk(logits, k, -1)

        recalls = get_recall(indices, targets)
        mrrs = get_mrr(indices, targets)
        #print("t",len(targets.cpu().numpy().tolist()),targets.cpu().numpy().tolist()[:10])

        ndcgs = ndcg_k(targets.cpu().numpy().tolist(),indices.cpu().numpy().tolist(), k)
        # print(indices.cpu().numpy().tolist(),len(indices.cpu().numpy().tolist()))

    return recalls, mrrs,ndcgs
