"""
@Project  : dichotomous-score
@File     : utils.py
@Author   : Shaobo Cui
@Date     : 08.09.2024 12:46
"""


# -*- coding: utf-8 -*-

import logging
from typing import List

import pandas as pd
from scipy import spatial
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AnglE')

def cosine_similarity(vec1: List[int], vec2: List[int]):
    """ Calculate cosine similarity between two vectors.

    :param vec1: a list of integers
    :param vec2: a list of integers
    :return: a float value between 0 and 1, indicating the similarity between the two vectors.
    """
    return 1 - spatial.distance.cosine(vec1, vec2)
