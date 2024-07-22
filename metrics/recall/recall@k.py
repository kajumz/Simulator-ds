from typing import List

#l = list(map(int, input().split()))
#s = list(map(float, input().split()))
#l = [1, 0, 0, 1, 1]
#s = [0.9, 0.8, 0.7, 0.6, 0.5]

def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """recall@k"""
    positive_samples = [index for index, sample in enumerate(labels) if sample == 1]
    total_positive_samples = len(positive_samples)

    sorted_predictions = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_k_predictions = sorted_predictions[:k]
    true_positive_samples = len(set(top_k_predictions).intersection(positive_samples))

    recall_at_k = true_positive_samples / total_positive_samples

    return recall_at_k
    #sor = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    #size = len(sor)
    #for i in range(size):
    #    if sor[i] > 0.5:
    #        sor[i] = 1
    #    else:
    #        sor[i] = 0
    #print(sor)
    #tp_rate = sor[:k]
    #fn_rate = sor[k:]
    #tp = 0
    #fn = 0
    #for (i, m), label in zip(enumerate(tp_rate), labels):
    #    if label == 1 and m == tp_rate[i]:
    #        tp += 1
    #for i, m in enumerate(fn_rate):
    #    if m == 1 and m != fn_rate[i]:
    #        fn += 1
    #print(tp)
    #print(fn)
    #if tp + fn == 0:
    #    return 0
    #else:
    #    return float(tp / (tp + fn))

def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """precision@k"""
    sorted_predictions = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_k_predictions = sorted_predictions[:k]
    true_positive_samples = sum([1 for index in top_k_predictions if labels[index] == 1])

    precision_at_k = true_positive_samples / k
    return precision_at_k
    #sor = sorted(scores, reverse=True)
    #size = len(sor)
    #for i in range(size):
    #    if sor[i] > 0.5:
    #        sor[i] = 1
    #    else:
    #        sor[i] = 0
    # print(sor)
    #tp_rate = sor[:k]
    #fn_rate = sor[k:]
    #tp = 0
    #fn = 0
    #for i, m in enumerate(tp_rate):
    #    if m == 1 and m == tp_rate[i]:
    #        tp += 1
    #for i, m in enumerate(fn_rate):
    #    if m == 1 and m != fn_rate[i]:
    #        fn += 1
    #print(tp)
    #print(fn)
    #if tp + fn == 0:
    #    return 0
    #else:
    #    return float(tp / (tp + fn))

def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """specificity@k"""
    negative_samples = [index for index, sample in enumerate(labels) if sample == 0]
    total_negative_samples = len(negative_samples)

    sorted_predictions = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_k_predictions = sorted_predictions[:k]
    true_negative_samples = len(set(top_k_predictions).intersection(negative_samples))
    if total_negative_samples == 0:
        return 0
    specificity_at_k = true_negative_samples / total_negative_samples
    return 1 - specificity_at_k
    #sor = sorted(scores, reverse=True)
    #size = len(sor)
    #for i in range(size):
    #    if sor[i] > 0.5:
    #        sor[i] = 1
    #    else:
    #        sor[i] = 0
    #fp_rate = sor[:k]
    #tp_rate = sor[k:]
    #fp = 0
    #tn = 0
    #for i, m in enumerate(labels):
    #    if m == 0 and m == sor[i]:
    #        tn += 1
    #    if m == 0 and m != sor[i]:
    #        fp += 1
    #    if m != 0:
    ##        continue
    #if tn + fp == 0:
    #    return 0
    #else:
    #    return tn / (tn+fp)



def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """recall@k"""
    rec = recall_at_k(labels, scores, k)
    pre = precision_at_k(labels, scores, k)
    if pre+rec == 0:
        return 0
    else:
        return 2*(pre*rec) / (pre+rec)

#print(specificity_at_k(l,s))