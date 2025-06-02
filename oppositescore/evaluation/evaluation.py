"""
@Project  : dichotomous-score
@File     : evaluation.py
@Author   : Shaobo Cui
@Date     : 10.09.2024 11:20
"""

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from typing import List
import numpy as np
from boltons.iterutils import chunked_iter
from tqdm import tqdm
from sklearn.metrics.pairwise import paired_cosine_distances
from oppositescore.model.base import AngleBase
import pandas as pd
import torch


class DichotomyEvaluator(object):
    def __init__(
        self,
        context: List[str],
        positive: List[str],
        negative: List[str],
        neutral: List[str],
        batch_size: int = 32
    ):
        assert len(context) == len(positive) == len(negative) == len(neutral), \
            "context, positive, negative, and neutral must have the same length"
        self.context = context
        self.positive = positive
        self.negative = negative
        self.neutral = neutral
        self.batch_size = batch_size

    def __call__(self, model: AngleBase, show_progress: bool = True, **kwargs) -> dict:
        """ Evaluate the model on the given dataset using DCF.
        :param model: DichotomyE, the model to evaluate.
        :param show_progress: bool, whether to show a progress bar during evaluation.
        :param kwargs: Additional keyword arguments to pass to the `encode` method of the model.
        :return: dict, The evaluation results including DCF.
        """
        # To store embeddings for context-aware combinations
        context_positive_embeddings = []
        context_negative_embeddings = []
        context_neutral_embeddings = []

        # Process data in batches
        for chunk in tqdm(chunked_iter(range(len(self.context)), self.batch_size),
                          total=len(self.context) // self.batch_size,
                          disable=not show_progress):
            # Concatenate context with positive, negative, and neutral respectively
            batch_context_positive = [self.context[i] + " " + self.positive[i] for i in chunk]
            batch_context_negative = [self.context[i] + " " + self.negative[i] for i in chunk]
            batch_context_neutral = [self.context[i] + " " + self.neutral[i] for i in chunk]

            # Encode the concatenated texts
            context_positive_embeddings.append(model.encode(batch_context_positive, **kwargs))
            context_negative_embeddings.append(model.encode(batch_context_negative, **kwargs))
            context_neutral_embeddings.append(model.encode(batch_context_neutral, **kwargs))

        # Concatenate the embeddings along the batch dimension
        context_positive_embeddings = np.concatenate(context_positive_embeddings, axis=0)
        context_negative_embeddings = np.concatenate(context_negative_embeddings, axis=0)
        context_neutral_embeddings = np.concatenate(context_neutral_embeddings, axis=0)

        # Calculate angles for DCF
        angles_pos_neutral = self._compute_angles(context_positive_embeddings, context_neutral_embeddings)
        angles_pos_neg = self._compute_angles(context_positive_embeddings, context_negative_embeddings)
        angles_neg_neutral = self._compute_angles(context_negative_embeddings, context_neutral_embeddings)

        dichotomy_score_pos_neutral = 1 - np.cos(angles_pos_neutral)
        dichotomy_score_pos_neg = 1 - np.cos(angles_pos_neg)
        dichotomy_score_neg_neutral = 1 - np.cos(angles_neg_neutral)

        # correlation = np.corrcoef(neutral_degree, self.neutral_prob)[0, 1]

        # print(f'angles_pos_neg: {angles_pos_neg}, angles_pos_neutral: {angles_pos_neutral}, angles_neg_neutral: {angles_neg_neutral}')
        angles_df = pd.DataFrame({
            'angles_pos_neutral': angles_pos_neutral,
            'angles_pos_neg': angles_pos_neg,
            'angles_neg_neutral': angles_neg_neutral
        })
        angles_df.to_csv('angles_data.csv', index=False)
        dcf_count = 0
        dcf_positive_count = 0
        dcf_negative_count = 0
        for i in range(len(angles_pos_neutral)):
            if (angles_pos_neutral[i] < angles_pos_neg[i]) and (angles_neg_neutral[i] < angles_pos_neg[i]):
                dcf_count += 1
            if (angles_pos_neutral[i] < angles_pos_neg[i]):
                dcf_positive_count += 1
            if (angles_neg_neutral[i] < angles_pos_neg[i]):
                dcf_negative_count += 1
        dcf = dcf_count / len(angles_pos_neutral)
        dcf_p = dcf_positive_count / len(angles_pos_neutral)
        dcf_n = dcf_negative_count / len(angles_pos_neutral)

        metrics = {
            "DCF": dcf,
            "DCF-positive": dcf_p,
            "DCF-negative": dcf_n,
            "pos_neg_degree":dichotomy_score_pos_neg.mean(),
            "pos_neutral_degree":dichotomy_score_pos_neutral.mean(),
            "neg_neutral_degree":dichotomy_score_neg_neutral.mean(),
        }

        return metrics

    @staticmethod
    def _compute_angles(embeddings1, embeddings2):
        """ Compute angles between pairs of embeddings.
        :param embeddings1: np.array, embeddings of the first set.
        :param embeddings2: np.array, embeddings of the second set.
        :return: np.array, angles between the embeddings.
        """
        angles = paired_cosine_distances(embeddings1, embeddings2)
        # print(cos_sim)
        # To calculate angles from cosine similarity: angle = arccos(cos_sim)
        return angles

    def _compute_complex_angles(self, embedding1, embedding2, pooling_strategy='sum'):
        """
        Compute angle loss
    
        :param y_true: torch.Tensor, ground truth.
            The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
        :param y_pred: torch.Tensor, model output.
            The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
        :param tau: float, scale factor, default 1.0
    
        :return: torch.Tensor, loss value
        """  # NOQA

        a, b = torch.chunk(torch.Tensor(embedding1), chunks=2, dim=1)
        c, d = torch.chunk(torch.Tensor(embedding2), chunks=2, dim=1)

        # (a+bi) / (c+di)
        # = ((a+bi) * (c-di)) / ((c+di) * (c-di))
        # = ((ac + bd) + i(bc - ad)) / (c^2 + d^2)
        # = (ac + bd) / (c^2 + d^2) + i(bc - ad)/(c^2 + d^2)
        z = torch.sum(c ** 2 + d ** 2, dim=1, keepdim=True)
        re = (a * c + b * d) / z
        im = (b * c - a * d) / z

        dz = torch.sum(a ** 2 + b ** 2, dim=1, keepdim=True) ** 0.5
        dw = torch.sum(c ** 2 + d ** 2, dim=1, keepdim=True) ** 0.5
        re /= (dz / dw)
        im /= (dz / dw)

        y_pred = torch.concat((re, im), dim=1)
        if pooling_strategy == 'sum':
            pooling = torch.sum(y_pred, dim=1)
        elif pooling_strategy == 'mean':
            pooling = torch.mean(y_pred, dim=1)
        else:
            raise ValueError(f'Unsupported pooling strategy: {pooling_strategy}')
        angle = (torch.abs(pooling))
        return angle
    # * tau)  # absolute delta angle
    # y_pred = y_pred[:, None] - y_pred[None, :]
    # y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    # zero = torch.Tensor([0]).to(y_pred.device)
    # y_pred = torch.concat((zero, y_pred), dim=0)
    # return torch.logsumexp(y_pred, dim=0)
