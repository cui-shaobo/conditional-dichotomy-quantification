"""
@Project  : dichotomous-score
@File     : loss.py
@Author   : Shaobo Cui
@Date     : 09.09.2024 14:01
"""
from typing import Optional

import torch
from torch import nn

# from oppositescore.model import cosine_loss, in_batch_negative_loss, angle_loss, contrastive_with_negative_loss
from oppositescore.model.angle import contrastive_with_negative_loss, cosine_loss, angle_loss, in_batch_negative_loss
from oppositescore.trainer.dichotomy_dataset import DatasetFormats



# def contrastive_loss_for_dichotomy(
#         partA: torch.Tensor,
#         partB: torch.Tensor,
#         tau: float = 20.0) -> torch.Tensor:
#     """
#     Compute contrastive with negative loss

#     :param partA:
#     :param partB:
#     :param tau: float, scale factor, default 20.0

#     :return: torch.Tensor, loss value
#     """
#     positive_norm = torch.nn.functional.normalize(partA, p=2, dim=1)  # (B, D)
#     negative_norm = torch.nn.functional.normalize(partB, p=2, dim=1)  # (B,D)
#     scores = torch.mm(positive_norm, negative_norm.transpose(0, 1))   # (B, B)
#     scores = (1 - scores) * tau
#     # print('scores: {}'.format(scores))
#     labels = torch.tensor(
#         range(len(scores)), dtype=torch.long, device=scores.device
#     )
#     return nn.CrossEntropyLoss()(scores, labels)
def contrastive_loss_for_dichotomy(positive: torch.Tensor, negative: torch.Tensor, neutral: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    positive_norm = torch.nn.functional.normalize(positive, p=2, dim=1)  # (B, D)
    negative_norm = torch.nn.functional.normalize(negative, p=2, dim=1)  # (B, D)
    neutral_norm = torch.nn.functional.normalize(neutral, p=2, dim=1)    # (B, D)

    pos_neg_similarity = torch.mm(positive_norm, negative_norm.transpose(0, 1))  # (B, B)
    pos_neg_similarity = (1.0 - pos_neg_similarity) * tau

    pos_neutral_similarity = torch.mm(positive_norm, neutral_norm.transpose(0, 1))  # (B, B)
    pos_neutral_similarity = (1.0 - pos_neutral_similarity) * tau 

    new_matrix = pos_neutral_similarity.clone() 
    new_matrix.fill_diagonal_(0)
    new_matrix += torch.diag(pos_neg_similarity.diagonal())
    labels = torch.tensor(range(len(new_matrix)), dtype=torch.long, device=new_matrix.device)
    return nn.CrossEntropyLoss()(new_matrix, labels)

class DichotomyLoss:
    """
    Configure DichotomyLoss.

    :param cosine_w: float. weight for cosine_loss. Default 1.0
    :param ibn_w: float. weight for contrastive loss. Default 1.0
    :param angle_w: float. weight for angle loss. Default 1.0
    :param cosine_tau: float. tau for cosine loss. Default 20.0
    :param ibn_tau: float. tau for contrastive loss. Default 20.0
    :param angle_tau: float. tau for angle loss. Default 20.0
    :param angle_pooling_strategy: str. pooling strategy for angle loss. Default'sum'.
    :param dataset_format: Optional[str]. Default None.
    """

    def __init__(self,
                 cosine_w: float = 0.0,
                 ibn_w: float = 20.0,
                 angle_w: float = 1.0,
                 dichotomy_w: float = 1.0,
                 dichotomy_contrastive_w: float = 1.0,
                 cosine_tau: float = 20.0,
                 ibn_tau: float = 20.0,
                 angle_tau: float = 20.0,
                 dichotomy_tau: float = 20.0,
                 dichotomy_contrastive_tau: float = 20.0,
                 angle_pooling_strategy: str = 'sum',
                 dataset_format: Optional[str] = None,
                 **kwargs):
        # if 'w1' in kwargs or 'w2' in kwargs or 'w3' in kwargs:
        #     assert ('w1, w2, and w3 has been renamed to cosine_w, ibn_w, and angle_w, respecitvely.'
        #             'Please use new names instead.')
        # For weights.
        self.cosine_w = cosine_w
        self.ibn_w = ibn_w
        self.angle_w = angle_w
        self.dichotomy_w = dichotomy_w
        self.dichotomy_contrastive_w = dichotomy_contrastive_w

        # For hyperparameter.
        self.cosine_tau = cosine_tau
        self.ibn_tau = ibn_tau
        self.angle_tau = angle_tau
        self.dichotomy_tau = dichotomy_tau
        self.dichotomy_contrastive_tau = dichotomy_contrastive_tau

        self.angle_pooling_strategy = angle_pooling_strategy
        self.dataset_format = dataset_format

    def __call__(self,
                 labels: torch.Tensor,
                 outputs: torch.Tensor) -> torch.Tensor:
        """ Compute loss for AnglE.

        :param labels: torch.Tensor. Labels.
        :param outputs: torch.Tensor. Outputs.

        :return: torch.Tensor. Loss.
        """
        if self.dataset_format == DatasetFormats.A:
            loss = 0.
            if self.cosine_w > 0:
                loss += self.cosine_w * cosine_loss(labels, outputs, self.cosine_tau)
            if self.ibn_w > 0:
                loss += self.ibn_w * in_batch_negative_loss(labels, outputs, self.ibn_tau)
            if self.angle_w > 0:
                loss += self.angle_w * angle_loss(labels, outputs, self.angle_tau,
                                                  pooling_strategy=self.angle_pooling_strategy)
        elif self.dataset_format == DatasetFormats.B:
            # text,positive,negative
            text = outputs[::3]
            positive = outputs[1::3]
            negative = outputs[2::3]
            assert text.shape == positive.shape == negative.shape, f'text.shape={text.shape}, postive.shape={positive.shape}, negative.shape={negative.shape}'  # NOQA

            _, fea_dim = text.shape
            positive_inputs = torch.stack((text, positive), dim=1).reshape(-1, fea_dim)  # zip(text, positive)
            positive_labels = torch.ones_like(positive_inputs[:, :1]).long()
            negative_inputs = torch.stack((text, negative), dim=1).reshape(-1, fea_dim)  # zip(text, negative)
            negative_labels = torch.zeros_like(negative_inputs[:, :1]).long()
            combined_inputs = torch.cat((positive_inputs, negative_inputs), dim=0)
            combined_labels = torch.cat((positive_labels, negative_labels), dim=0)

            loss = 0.
            if self.cosine_w > 0:
                loss += self.cosine_w * cosine_loss(combined_labels, combined_inputs, self.cosine_tau)
            if self.ibn_w > 0:
                loss += self.ibn_w * contrastive_with_negative_loss(text, positive, negative, tau=self.ibn_tau)
            if self.angle_w > 0:
                loss += self.angle_w * angle_loss(combined_labels, combined_inputs, self.angle_tau,
                                                  pooling_strategy=self.angle_pooling_strategy)
        elif self.dataset_format == DatasetFormats.C:
            text = outputs[::2]
            positive = outputs[1::2]
            loss = contrastive_with_negative_loss(text, positive, neg=None, tau=self.ibn_tau)
        elif self.dataset_format == DatasetFormats.D:
            # context, positive, negative, neutral
            # context = outputs[::4]
            context_positive = outputs[0::3]
            context_negative = outputs[1::3]
            context_neutral = outputs[2::3]
            assert context_positive.shape == context_negative.shape == context_neutral.shape, f'context_positive.shape={context_positive.shape}, context_negative.shape={context_negative.shape},  context_neutral.shape={context_neutral.shape}'

            # For now, what is plus vs. minus pair:
            # (i) [context_positive, context_neutral] vs. [context_positive, context_negative]
            # (ii) [context_negative, context_neutral] vs. [context_positive, context_negative]

            # For (i) [context_positive, context_neutral] vs. [context_positive, context_negative]
            _, fea_dim = context_positive.shape

            ##################################
            # Take the positive as the origin.
            ##################################
            plus_positive_neutral_inputs = torch.stack((context_positive, context_neutral), dim=1).reshape(-1, fea_dim)
            plus_positive_neutral_labels = torch.ones_like(plus_positive_neutral_inputs[:, :1]).long()

            minus_positive_negative_inputs = torch.stack((context_positive, context_negative), dim=1).reshape(-1, fea_dim)
            minus_positive_negative_labels = torch.zeros_like(minus_positive_negative_inputs[:, :1]).long()

            combined_inputs_positive_origin = torch.cat((plus_positive_neutral_inputs, minus_positive_negative_inputs),
                                                        dim=0)
            combined_labels_positive_origin = torch.cat((plus_positive_neutral_labels, minus_positive_negative_labels),
                                                        dim=0)

            # Take the negative as the origin.
            plus_negative_neutral_inputs = torch.stack((context_negative, context_neutral), dim=1).reshape(-1, fea_dim)
            # print('plus_negative_neutral_inputs.size() {}'.format(plus_negative_neutral_inputs.size()))
            """
            [
              [context_negative[0]],  # Row 1
              [context_neutral[0]],   # Row 2
              [context_negative[1]],  # Row 3
              [context_neutral[1]],   # Row 4
              ...
              [context_negative[7]],  # Row 15
              [context_neutral[7]]    # Row 16
            ]
            """
            plus_negative_neutral_labels = torch.ones_like(plus_negative_neutral_inputs[:, :1]).long()
            minus_negative_positive_inputs = torch.stack((context_negative, context_positive), dim=1).reshape(-1,
                                                                                                              fea_dim)
            minus_negative_positive_labels = torch.zeros_like(minus_negative_positive_inputs[:, :1]).long()

            combined_inputs_negative_origin = torch.cat((plus_negative_neutral_inputs, minus_negative_positive_inputs),
                                                        dim=0)
            combined_labels_negative_origin = torch.cat((plus_negative_neutral_labels, minus_negative_positive_labels),
                                                        dim=0)

            # For loss function
            loss = 0.

            # For dichotomy loss.
            if self.dichotomy_w > 0:
                # Take the positive as the origin.
                # dichotomy_loss_positive_origin = angle_loss(combined_labels_positive_origin,
                #                                                                combined_inputs_positive_origin,
                #                                                                self.dichotomy_tau,
                #                                                                pooling_strategy=self.angle_pooling_strategy)
                # # print(combined_labels_negative_origin, combined_inputs_positive_origin.size())
                # dichotomy_loss_negative_origin = angle_loss(combined_labels_negative_origin,
                #                                                                combined_inputs_negative_origin,
                #                                                                self.dichotomy_tau,
                #                                                                pooling_strategy=self.angle_pooling_strategy)
                # main_dichotomy_loss = dichotomy_loss_positive_origin + dichotomy_loss_negative_origin
                # # print(f'dichotomy_loss \npositive: {dichotomy_loss_positive_origin}, negative: {dichotomy_loss_negative_origin}, overall: {main_dichotomy_loss}')
                # loss += (self.dichotomy_w * main_dichotomy_loss)
                dichotomy_loss=angle_loss(context_positive,context_neutral,context_negative,self.dichotomy_tau,pooling_strategy=self.angle_pooling_strategy)
                loss += (self.dichotomy_w * dichotomy_loss)

            # For contrastive loss for dichotomy.
            # if self.contrastive_learning_dichotomy_w > 0:
            if self.dichotomy_contrastive_w > 0:
                positive_origin_contrastive_dichotomy_loss = contrastive_loss_for_dichotomy(positive=context_positive,
                                                                                            negative=context_negative,
                                                                                            neutral=context_neutral,
                                                                                            tau=self.dichotomy_contrastive_tau)
                negative_origin_contrastive_dichotomy_loss = contrastive_loss_for_dichotomy(positive=context_negative,
                                                                                            negative=context_positive,
                                                                                            neutral=context_neutral,
                                                                                            tau=self.dichotomy_contrastive_tau)
                contrastive_dichotomy_loss = positive_origin_contrastive_dichotomy_loss + negative_origin_contrastive_dichotomy_loss
                loss += (self.dichotomy_contrastive_w * contrastive_dichotomy_loss)
                # print(f'dichotomy_contrastive_loss \npositive: {positive_origin_contrastive_dichotomy_loss}, negative: {negative_origin_contrastive_dichotomy_loss}, overall: {contrastive_dichotomy_loss}')
                # print(f'main loss: {loss}')


        else:
            raise NotImplementedError
        return loss
