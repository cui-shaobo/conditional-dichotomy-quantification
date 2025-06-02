"""
@Project  : dichotomous-score
@File     : dichotomye_trainer.py
@Author   : Shaobo Cui
@Date     : 08.09.2024 16:11
"""
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, AutoModel


from oppositescore.model.angle import Pooler, DatasetFormats, check_llm, get_pooling
from oppositescore.trainer.loss import DichotomyLoss
from oppositescore.utils.utils import logger


class DichotomyTrainer(Trainer):
    """
    Custom Huggingface Trainer for DichotomyE.

    :param pooler: Pooler. Required
    :param loss_kwargs: Optional[Dict]. Default None.
    :param dataset_format: str. Default DatasetFormats.A
    :param teacher_name_or_path: Optional[str]. For distribution alignment.
    :param **kwargs: other parameters of Trainer.
    """

    def __init__(self,
                 pooler: Pooler,
                 loss_kwargs: Optional[Dict] = None,
                 dataset_format: str = DatasetFormats.A,
                 teacher_name_or_path: Optional[str] = None,
                 teacher_pooling_strategy: str = 'cls',
                 **kwargs):
        super().__init__(**kwargs)
        self.pooler = pooler
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_fct = DichotomyLoss(dataset_format=dataset_format, **loss_kwargs)
        self.teacher_name_or_path = teacher_name_or_path
        self.teacher_pooling_strategy = teacher_pooling_strategy
        if teacher_name_or_path is not None:
            logger.info('Teacher detected! '
                        'please ensure the teacher has the same tokenizer as the backbone model!')
            assert not check_llm(teacher_name_or_path), ('Currently not support LLMs alignment,'
                                                         f' teacher={teacher_name_or_path}')
            teacher_backbone = AutoModel.from_pretrained(
                teacher_name_or_path,
                trust_remote_code=True,
                torch_dtype=self.pooler.model.dtype).to(self.pooler.model.device)

            self.teacher_pooler = Pooler(
                teacher_backbone,
                pooling_strategy=self.teacher_pooling_strategy,
                padding_strategy=self.pooler.padding_strategy)
            logger.info(f'Train with teacher={teacher_name_or_path}')

    def distillation_loss(self,
                          inputs: torch.Tensor,
                          targets: torch.Tensor,
                          mse_weight: float = 1.0,
                          kl_temperature: float = 1.0) -> torch.Tensor:
        """ Compute distillation loss.

        :param inputs: torch.Tensor. Input tensor.
        :param targets: torch.Tensor. Target tensor.
        :param mse_weight: float. MSE weight. Default 1.0.
        :param kl_temperature: float. KL temperature. Default 1.0.

        :return: torch.Tensor. Distillation loss.
        """
        loss = 0.
        if mse_weight > 0:
            loss += mse_weight * nn.MSELoss()(inputs, targets)
        if kl_temperature > 0:
            loss += nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(inputs / kl_temperature, dim=-1),
                F.softmax(targets / kl_temperature, dim=-1)
            ) * kl_temperature
        return loss

    # Rewrite the compute_loss for training dichotomy embedding.
    def compute_loss(self, model, inputs, return_outputs=False):
        """ Compute loss for AnglE.

        :param model: Huggingface model.
        :param inputs: Dict. Model inputs.
        :param return_outputs: bool. Return outputs or not. Default False.

        :return: torch.Tensor. Loss.
        """
        labels = inputs.pop("labels", None)
        if self.teacher_name_or_path is not None:
            all_outputs = self.pooler(inputs, layer_index=-1, return_all_layer_outputs=True)[-1]
            outputs = get_pooling(all_outputs, inputs,
                                  self.pooler.pooling_strategy,
                                  self.pooler.padding_strategy)
            loss = self.loss_fct(labels, outputs)
            with torch.no_grad():
                self.teacher_pooler.model = self.teacher_pooler.model.to(self.pooler.model.device)
                align_outputs = self.teacher_pooler(inputs)

            alignment_loss = self.distillation_loss(
                all_outputs if self.teacher_pooling_strategy == 'all' else outputs,
                align_outputs,
                mse_weight=0.0,
                kl_temperature=1.0)
            loss += alignment_loss
        else:
            outputs = self.pooler(inputs)
            loss = self.loss_fct(labels, outputs)
        # print("#" * 20, labels)
        loss.requires_grad = True
        return (loss, outputs) if return_outputs else loss

