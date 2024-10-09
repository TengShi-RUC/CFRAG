import sys

sys.path.append(".")

import math

import evaluate
import jieba
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_chinese import Rouge
from tqdm import tqdm


class LaMPEvaluation:

    def __init__(self, task):
        self.task = task
        if task.startswith('LaMP_1'):
            self.metric = Metric_F1_Accuracy(self._get_labels(task))
        elif task.startswith('LaMP_2'):
            self.metric = Metric_F1_Accuracy(self._get_labels(task))
        elif task.startswith('LaMP_3'):
            self.metric = Metric_MAE_RMSE()
        else:
            self.metric = Metric_BLEU_ROUGE()

    def compute_metrics(self, preds, labels, avg=True):
        return self.metric.compute_metrics(preds, labels, avg)

    def _get_labels(self, task_name):
        if task_name.startswith("LaMP_1"):
            return ["[1]", "[2]"]
        elif task_name.startswith("LaMP_2"):
            return [
                'sci-fi', 'based on a book', 'comedy', 'action',
                'twist ending', 'dystopia', 'dark comedy', 'classic',
                'psychology', 'fantasy', 'romance', 'thought-provoking',
                'social commentary', 'violence', 'true story'
            ]
        elif task_name.startswith("LaMP_3"):
            return ["1", "2", "3", "4", "5"]
        else:
            raise ValueError("Invalid task_name")


def postprocess_text_classification(preds, labels):
    preds = [str(pred).strip().lower() for pred in preds]
    labels = [str(label).strip().lower() for label in labels]
    return preds, labels


class Metric_F1_Accuracy:

    def __init__(self, all_labels):
        self.all_labels = all_labels
        self.f1_metric = evaluate.load("metrics/f1.py")
        self.accuracy_metric = evaluate.load("metrics/accuracy.py")

    def create_mapping(self, x):
        try:
            return self.all_labels.index(x)
        except:
            return -1

    def _compute_metrics(self, preds, labels):
        preds, labels = postprocess_text_classification(preds, labels)
        preds = [self.create_mapping(x) for x in preds]
        labels = [self.create_mapping(x) for x in labels]
        result_acc = self.accuracy_metric.compute(predictions=preds,
                                                  references=labels)
        result_f1 = self.f1_metric.compute(predictions=preds,
                                           references=labels,
                                           labels=list(
                                               range(len(self.all_labels))),
                                           average="macro")
        result = {"accuracy": result_acc["accuracy"], "f1": result_f1["f1"]}
        return result

    def compute_metrics(self, preds, labels, avg=True):
        if avg:
            return self._compute_metrics(preds, labels)
        else:
            results = {"accuracy": [], "f1": []}
            for pred, label in zip(preds, labels):
                cur_score = self._compute_metrics([pred], [label])
                for k in results.keys():
                    results[k].append(cur_score[k])

            return results


class Metric_MAE_RMSE:

    def __init__(self):
        self.mse_metric = evaluate.load("metrics/mse.py")
        self.mae_metric = evaluate.load("metrics/mae.py")

    def create_mapping(self, x, y):
        try:
            return float(x)
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0

    def _compute_metrics(self, preds, labels):
        preds, labels = postprocess_text_classification(preds, labels)
        preds = [self.create_mapping(x, y) for x, y in zip(preds, labels)]
        labels = [self.create_mapping(x, x) for x in labels]
        result_mae = self.mae_metric.compute(predictions=preds,
                                             references=labels)
        result_rmse = self.mse_metric.compute(predictions=preds,
                                              references=labels)
        result = {"MAE": result_mae["mae"], "MSE": result_rmse["mse"]}
        return result

    def compute_metrics(self, preds, labels, avg=True):
        if avg:
            results = self._compute_metrics(preds, labels)
            results['RMSE'] = math.sqrt(results['RMSE'])
        else:
            results = {"MAE": [], "MSE": []}
            for pred, label in zip(preds, labels):
                cur_score = self._compute_metrics([pred], [label])
                for k in results.keys():
                    results[k].append(cur_score[k])

            return results


class Metric_BLEU_ROUGE:

    def __init__(self):
        pass

    def compute_metrics(self, preds, labels, avg=True):
        results = compute_metric_bleu_rouge(preds, labels, avg)
        return results


def postprocess_text_generation(preds, labels):
    preds = [pred.strip().lower() for pred in preds]
    labels = [[label.strip().lower()] for label in labels]

    return preds, labels


def compute_metric_bleu_rouge(preds, labels, avg=True, use_tqdm=False):
    preds = [x.lower() for x in preds]
    labels = [x.lower() for x in labels]

    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    if use_tqdm:
        iterator = tqdm(zip(preds, labels))
    else:
        iterator = zip(preds, labels)
    for pred, label in iterator:
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))

        if len(" ".join(hypothesis).split()) == 0 or len(
                " ".join(reference).split()) == 0:
            result = {
                "rouge-1": {
                    "f": 0.0
                },
                "rouge-2": {
                    "f": 0.0
                },
                "rouge-l": {
                    "f": 0.0
                }
            }
        else:
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(hypothesis),
                                      " ".join(reference))
            result = scores[0]

        for k, v in result.items():
            score_dict[k].append(v["f"])

        bleu_score = sentence_bleu(
            [list(label)],
            list(pred),
            smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(bleu_score)

    if avg:
        return {k: float(np.mean(v)) for k, v in score_dict.items()}
    else:
        return score_dict
