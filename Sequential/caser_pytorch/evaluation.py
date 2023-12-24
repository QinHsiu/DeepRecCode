import numpy as np
import math

def _compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


def _compute_precision_recall(targets, predictions, k):
    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    #recall = float(num_hit) / len(targets)
    recall = float(num_hit)
    return precision, recall


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg

    return res #/ float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res



def evaluate_ranking(model, test, train=None, k=10):
    """
    Compute Precision@k, Recall@k scores and average precision (AP).
    One score is given for every user with interactions in the test
    set, representing the AP, Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, rated items in
        interactions will be excluded.
    k: int or array of int,
        The maximum number of predicted items
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    ndcgs=[list() for _ in range(len(ks))]
    apks = list()

    for user_id, row in enumerate(test):
        ndcg=0

        if not len(row.indices):
            continue
        # print("ui",user_id)
        predictions = -model.predict(user_id)
        predictions = predictions.argsort()
        #print("length",len(predictions))

        if train is not None:
            rated = set(train[user_id].indices)
        else:
            rated = []

        predictions = [p for p in predictions if p not in rated]
        #print("predictions",len(predictions))

        targets = [row.indices[-1]]
        #print("targets",targets[-1])
        #
        # if user_id==2:
        #     break
        # continue

        for i, _k in enumerate(ks):
            # print("_k",_k)
            precision, recall = _compute_precision_recall(targets, predictions, _k)
            # print("recall:",recall)
            precisions[i].append(precision)
            recalls[i].append(recall)
            t_k=1
            idcg=idcg_k(t_k)
            dcg_k = sum([int(predictions[j] in set(targets)) / math.log(j + 2, 2) for j in range(_k)])
            ndcg+=dcg_k/idcg
            ndcgs[i].append(ndcg)
        apks.append(_compute_apk(targets, predictions, k=np.inf))
    #print("metrics:",recalls)
    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]
    ndcgs=[np.array(i) for i in ndcgs]

    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]

    mean_aps = np.mean(apks)

    return precisions, recalls, mean_aps,ndcgs
