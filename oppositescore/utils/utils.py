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
from neutral.irrelevant_sentence_generator import IrrelevantSentenceGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AnglE')

def evaluation_results_latex_format_convert(evaluation_results: dict, output_columns: list, decimal_places: int = 2):
    # Convert the evaluation results dictionary to a DataFrame
    df = pd.DataFrame([evaluation_results])  # Wrapping in a list to create a single-row DataFrame

    # Select the specified output columns
    output_metrics = df[output_columns]

    # Multiply by 100 and round to the specified decimal places
    output_metrics = (output_metrics * 100).round(decimal_places)

    # Convert each metric to string and join with "&"
    output_format = " & ".join(output_metrics.astype(str).values.flatten())

    return output_format

def filter_irrelevant_sentences(contexts_and_irrelevant_sentences, batch_size=32):
    """
    run the nli model and fill the nli_true_prob and sub_questions_nli_true_prob fields
    adapted from https://github.com/oriyor/ret-robust/blob/main/nli/src/utils.py
    """
    # nli model
    device = "cuda"
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-large-mnli"
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    nli_model.to(device)
    nli_model.eval()

    contexts_and_irrelevant_sentences_new = []
    # Batch processing
    for i in tqdm(range(0, len(contexts_and_irrelevant_sentences), batch_size)):
        batch = contexts_and_irrelevant_sentences[i : min(i + batch_size, len(contexts_and_irrelevant_sentences))]

        with torch.no_grad():  # Disable autograd to save memory
            # run through model pre-trained on MNLI
            # we compute the probability of two sentences being neutral bidirectionally
            for hypo in ["Sentence 1 is neutral to Sentence 2.", "Sentence 2 is neutral to Sentence 1."]:
                batch_encoded = tokenizer.batch_encode_plus(
                    [(
                        f"Sentence 1: {example['neutral_to']} Sentence 2: {example['neutral']}",
                        hypo
                    ) for example in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation_strategy="only_first",
                ).to(device)
                logits = nli_model(**batch_encoded)[0]

                # we throw away "neutral" (dim 1) and take the probability of
                # "entailment" (2) as the probability of the label being true 
                entail_contradiction_logits = logits[:, [0, 2]]
                probs = entail_contradiction_logits.softmax(dim=1)
                prob_label_is_true = probs[:, 1]
                for j, _ in enumerate(batch):
                    try:
                        contexts_and_irrelevant_sentences[i+j]["neutral_prob"] += 0.5*prob_label_is_true[j].item()
                    except:
                        contexts_and_irrelevant_sentences[i+j]["neutral_prob"] = 0.5*prob_label_is_true[j].item()

        torch.cuda.empty_cache()  # Clear GPU memory after processing

        for example in batch:
            for si, positive in example["positives"]:
                for di, negative in example["negatives"]:
                    # if si in example["neutral_id"] or di in example["neutral_id"]:
                    # -_-! Quite tricky bugs.
                    si_sub = si + '_'
                    di_sub = di + '_'
                    if si_sub in example["neutral_id"] or di_sub in example["neutral_id"]:
                        contexts_and_irrelevant_sentences_new.append({
                            "context": example["context"],
                            "context_id": example["context_id"],
                            "positive": positive,
                            "positive_id": si,
                            "negative": negative,
                            "negative_id": di,
                            "neutral": example["neutral"],
                            "neutral_id": example["neutral_id"],
                            "neutral_to": example["neutral_to"],
                            "irrelevant_sentence_mode": example["irrelevant_sentence_mode"],
                            "neutral_prob": example["neutral_prob"]
                        })

    return contexts_and_irrelevant_sentences_new


def generate_irrelevant_sentences(contexts_and_irrelevant_sentences, data, name, gpt_model, num_noun_chunk):
    # Load OpenAI API key
    with open("api_key/config.json") as f:
        OPENAI_API_KEY = json.load(f)["openai_api_key"]

    # Initialize the irrelevant sentence generator
    irrelevant_sentence_generator = IrrelevantSentenceGenerator(
        model_name=gpt_model,
        api_key=OPENAI_API_KEY
    )

    for example in tqdm(data):
        # Define the context, the sentence that the irrelevant sentence should be irrelevant to, and dichotomous sentences
        if name == "defeasible_snli":
            if example['cause'][-1] != '.':
                example['cause'] += '.'
            if example['long_term_effect'][-1] != '.':
                example['long_term_effect'] += '.'
            context = f"Premise: {example['cause']} Hypothesis: {example['long_term_effect']}"
            hypothesis = example['long_term_effect']
            positives = example['assumptions']
            negatives = example['defeaters']
        elif name == "delta_causal":
            if example['cause'][-1] != '.':
                example['cause'] += '.'
            if example['long_term_effect'][-1] != '.':
                example['long_term_effect'] += '.'
            context = f"Cause: {example['cause']} Effect: {example['long_term_effect']}"
            hypothesis=example["cause"]
            positives = example['assumptions']
            negatives = example['defeaters']
        elif name == "perspectrum":
            if example['claim_text'][-1] != '.':
                example['claim_text'] += '.'
            context = f"Claim: {example['claim_text']}"
            hypothesis = example['claim_text']
            positives = example['supporters']
            negatives = example['opposers']
            
        # # Generate irrelevant sentences

        positives = [(f"c{example['id']}_s{si}", positive) for si, positive in enumerate(positives)]
        negatives = [(f"c{example['id']}_d{di}", negative) for di, negative in enumerate(negatives)]
        for ni, (si_or_di, sentence) in enumerate(positives+negatives):
            # print(sentence)
            # print(positives)
            irrelevant_sentences = irrelevant_sentence_generator.generate_irrelevant_sentence(context, sentence, dataset_name=name, num_noun_chunk=num_noun_chunk)
            for mode, sents in irrelevant_sentences.items():
                for sent in sents:
                    print('#' * 50 + '\n' + sent)
                    # Follow the structure of DatasetFormats.D in the paper
                    contexts_and_irrelevant_sentences.append(
                        {
                            "context": context,
                            "context_id": f"c{example['id']}",
                            "positives": positives,
                            "negatives": negatives,
                            "neutral": sent,
                            "neutral_id": f"{si_or_di}_n{ni}_p3",
                            "neutral_to": hypothesis,
                            "irrelevant_sentence_mode": mode, 
                            "num_noun_chunk": num_noun_chunk, 
                            "gpt_model": gpt_model
                        }
                    )
                    print(contexts_and_irrelevant_sentences[-1])
            # for positive in positives:
            #     for negative in negatives:
                    # for mode, sents in irrelevant_sentences.items():
                    #     for sent in sents:
                    #         # Follow the structure of DatasetFormats.D in the paper
                    #         contexts_and_irrelevant_sentences.append(
                    #             {
                    #                 "context": context,
                    #                 "positive": positive,
                    #                 "negative": negative,
                    #                 "neutral": sent,
                    #                 "neutral_to": hypothesis,
                    #                 "irrelevant_sentence_mode": mode
                    #             }
                    #         )

    return contexts_and_irrelevant_sentences


def cosine_similarity(vec1: List[int], vec2: List[int]):
    """ Calculate cosine similarity between two vectors.

    :param vec1: a list of integers
    :param vec2: a list of integers
    :return: a float value between 0 and 1, indicating the similarity between the two vectors.
    """
    return 1 - spatial.distance.cosine(vec1, vec2)
