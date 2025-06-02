"""
@Project  : dichotomous-score
@File     : dichotomyInferencer.py
@Author   : Shaobo Cui
@Date     : 17.10.2024 20:42
"""


from typing import List
import numpy as np
from boltons.iterutils import chunked_iter
from tqdm import tqdm
from sklearn.metrics.pairwise import paired_cosine_distances
from oppositescore.model.base import AngleBase


class DichotomyInferencer(object):
    def __init__(
        self,
        context: List[str],
        sent1: List[str],
        sent2: List[str],
        batch_size: int = 1
    ):
        assert len(context) == len(sent1) == len(sent2), \
            "context, positive, negative, and neutral must have the same length"
        self.context = context
        self.sent1 = sent1
        self.sent2 = sent2
        self.batch_size = batch_size

    def __call__(self, model: AngleBase, show_progress: bool = True, **kwargs) -> dict:
        """ Evaluate the model on the given dataset using DCF.
        :param model: DichotomyE, the model to evaluate.
        :param show_progress: bool, whether to show a progress bar during evaluation.
        :param kwargs: Additional keyword arguments to pass to the `encode` method of the model.
        :return: dict, The evaluation results including DCF.
        """
        # To store embeddings for context-aware combinations
        context_sent1_embeddings = []
        context_sent2_embeddings = []

        # Process data in batches
        for chunk in tqdm(chunked_iter(range(len(self.context)), self.batch_size),
                          total=len(self.context) // self.batch_size,
                          disable=not show_progress):
            # Concatenate context with positive, negative, and neutral respectively
            batch_context_sent1 = [self.context[i] + " " + self.sent1[i] for i in chunk]
            batch_context_sent2 = [self.context[i] + " " + self.sent2[i] for i in chunk]

            # Encode the concatenated texts
            context_sent1_embeddings.append(model.encode(batch_context_sent1, **kwargs))
            context_sent2_embeddings.append(model.encode(batch_context_sent2, **kwargs))

        # Concatenate the embeddings along the batch dimension
        context_sent1_embeddings = np.concatenate(context_sent1_embeddings, axis=0)
        context_sent2_embeddings = np.concatenate(context_sent2_embeddings, axis=0)

        # Calculate angles for DCF
        angles_pos_neg = self._compute_angles(context_sent1_embeddings, context_sent2_embeddings)

        return angles_pos_neg

    @staticmethod
    def _compute_angles(embeddings1, embeddings2):
        """ Compute angles between pairs of embeddings.
        :param embeddings1: np.array, embeddings of the first set.
        :param embeddings2: np.array, embeddings of the second set.
        :return: np.array, angles between the embeddings.
        """
        angles = paired_cosine_distances(embeddings1, embeddings2)
        # print(cos_sim)
        return angles