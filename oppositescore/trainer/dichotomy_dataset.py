"""
@Project  : dichotomous-score
@File     : dichotomy_dataset.py
@Author   : Shaobo Cui
@Date     : 08.09.2024 19:21
"""
import re

from typing import Optional, List, Dict, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from oppositescore.utils.utils import logger


class DatasetFormats:
    """
    Predefined Data Formats.

    Check all available formats:

            from angle_emb import DatasetFormats

            print(DatasetFormats.list_formats())

    """

    """
    format A: text1,text2,label
    input format: [
        text1[0],
        text2[0],
        text1[1],
        text2[1],
        ...
    ]
    label format: [
        label[0],
        label[0],
        label[1],
        label[1],
        ...
    ]
    """
    A = 'text1,text2,label'

    """
    format B: text,positive,negative
    input format: [
        text[0],
        positive[0],
        negative[0],
        text[1],
        positive[1],
        negative[1],
        ...
    ]
    """
    B = 'text,positive,negative'

    """
    format C: text,positive
    input format: [
        text[0],
        positive[0],
        text[1],
        positive[1],
        ...
    ]
    """
    C = 'text,positive'

    """
    format D: context_text,supporter_text,defeater_text,neutral_text
    input format: [
        context_text[0],
        supporter_text[0],
        defeater_text[0],
        neutral_text[0],
        context_text[1],
        supporter_text[1],
        defeater_text[1],
        neutral_text[1],
        ...
    ]
    """
    D = 'context_text,supporter_text,defeater_text,neutral_text'


    @classmethod
    def list_formats(cls):
        for key, val in DatasetFormats.__dict__.items():
            if key.startswith('_') or key == 'list_formats':
                continue
            print(f'DatasetFormats.{key}', '=', f"'{val}'")


class DichotomyEDataTokenizer:
    """
    Tokenize data using DichotomyEDataTokenizer.

    :param tokenizer: PreTrainedTokenizerBase. Tokenizer
    :param max_length: Optional[int]. Specify max length
    :param prompt_template: Optional[str], set prompt template, it will be applied to all input texts. Default None
    :param extra_columns: Optional[List[str]].
        If providing multiple placeholders in prompt_template, specify their name via extra_columns. Default None
    :param dataset_format: Optional[str]. Specify dataset_format from DatasetFormats. Default None.
        It will automatically detect the dataset format.
    :param end_with_eos: bool. Specify whether ends with the eos token. Default False.

    Example::

            from data import load_dataset
            from dichotomye import DichotomyE
            for dichotomy_dataset import DichotomyEDataTokenizer

            # define dataset
            ds = load_dataset('your_dataset')
            # define angle
            dichotomye = DichotomyE(*args, **kwargs)
            # tokenize data
            train_ds = ds['train'].shuffle().map(DichotomyEDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
            valid_ds = ds['validation'].map(DichotomyEDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)

    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: Optional[int] = None,
                 prompt_template: Optional[str] = None,
                 template_placeholders: Optional[List[str]] = None,
                 extra_columns: Optional[List[str]] = None,
                 dataset_format: Optional[str] = None,
                 end_with_eos: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.prompt_template_tok = None
        self.extra_columns = extra_columns
        self.dataset_format = dataset_format
        self.end_with_eos = end_with_eos
        if template_placeholders is None:
            template_placeholders = ['condition', 'text']
        if prompt_template is not None:
            re_placeholder = re.compile(r'\{(%s)\}' % '|'.join(template_placeholders))
            self.prompt_template_tok = self.tokenizer(re_placeholder.sub('', prompt_template))

    @staticmethod
    def fix_bad_data(token_ids, prompt_ids):
        bad_index = -1
        for idx in range(len(token_ids) - 1, -1, -1):
            try:
                bad_index = prompt_ids.index(token_ids[idx])
            except ValueError:
                break
        if bad_index == -1:
            return token_ids
        # print('bad index:', prompt_ids[bad_index])
        to_fix_ids = prompt_ids[bad_index:]
        return token_ids[:len(token_ids) - len(to_fix_ids)] + to_fix_ids

    def __call__(self, data: Dict) -> Dict:
        # print(data)
        # Detect the format based on column names.
        if self.dataset_format is None:
            if 'text1' in data and 'text2' in data and 'label' in data:
                logger.info(f'Detect DatasetFormats.A: {DatasetFormats.A}')
                self.dataset_format = DatasetFormats.A
            elif 'text' in data and 'positive' in data and 'negative' in data:
                self.dataset_format = DatasetFormats.B
                logger.info(f'Detect DatasetFormats.B: {DatasetFormats.B}')
            elif 'text' in data and 'positive' in data and 'negative' not in data and 'label' not in data:
                self.dataset_format = DatasetFormats.C
                logger.info(f'Detect DatasetFormats.C: {DatasetFormats.C}')
            elif 'context' in data and 'positive' in data and 'negative' in data and 'neutral' in data:
                self.dataset_format = DatasetFormats.D
                logger.info(f'Detect DatasetFormats.D: {DatasetFormats.D}')
            else:
                raise NotImplementedError('Currently only support two dataset formats'
                                          'DatasetFormats A: must include three columns: `text1`, `text2`, and `label`.'
                                          'DatasetFormats B: must include three columns: `text`, `positive`, `negative`'
                                          'DatasetFormats C: must include three columns: `text`, `positive`'
                                          'DatasetFormats D: must include four columns: `context`, `positive`, `negative`, `neutral`')
        text_columns = None
        # print(self.dataset_format)
        if self.dataset_format == DatasetFormats.A:
            text_columns = ['text1', 'text2']
        elif self.dataset_format == DatasetFormats.B:
            text_columns = ['text', 'positive', 'negative']
        elif self.dataset_format == DatasetFormats.C:
            text_columns = ['text', 'positive']
        elif self.dataset_format == DatasetFormats.D:
            # text_columns = ['context', 'positive', 'negative', 'neutral']
            context = data['context']
            positive = data['positive']
            negative = data['negative']
            neutral = data['neutral']

            # Create combined text sequences.
            data['context_positive'] = f"{context} {positive}"
            data['context_negative'] = f"{context} {negative}"
            data['context_neutral'] = f"{context} {neutral}"

            # Set the text columns to include the new combined sequences.
            text_columns = ['context_positive', 'context_negative', 'context_neutral']

        # print(text_columns)
        # This is for handling different text columns.
        extra_length = 0
        extra_placeholder = {}
        if self.extra_columns is not None:
            for key, val in data.items():
                if key not in self.extra_columns:
                    continue
                extra_placeholder[key] = val
                extra_length += len(self.tokenizer(val, add_special_tokens=False)['input_ids'])
        if self.end_with_eos:
            extra_length += 1

        # Applying the prompt template.
        if self.prompt_template_tok is not None:
            # max_length is adjusted by subtracting the length of the tokenized prompt template and any extra length
            # contributed by additional tokens.
            max_length = self.max_length - len(self.prompt_template_tok['input_ids']) - extra_length #
            for text_column in text_columns:
                # The text is tokenized with the specified max_length and truncation enabled. Special tokens (like
                # [CLS] or [SEP]) are not added at this stage to keep the template clean.
                tok = self.tokenizer(data[text_column],
                                     max_length=max_length,
                                     truncation=True,
                                     add_special_tokens=False)
                data[text_column] = self.tokenizer.decode(tok['input_ids'])

                # Apply prompt template and replace back the original text_column.
                data[text_column] = self.prompt_template.format(text=data[text_column], **extra_placeholder)

        # This step tokenizes each text column individually and stores the resulting tokens in the toks list.
        toks = []
        for text_column in text_columns:
            toks.append(self.tokenizer(data[text_column], max_length=self.max_length, truncation=True))

        # This step ensures that the tokens from the prompt template match the tokens generated from the text data.
        if self.prompt_template_tok is not None:
            for tok in toks:
                if tok['input_ids'][-1] != self.prompt_template_tok['input_ids'][-1]:
                    logger.info(
                        f"data data: token ids={tok['input_ids']}, prompt_token_ids={self.prompt_template_tok['input_ids']}")  # NOQA
                    tok['input_ids'] = self.fix_bad_data(tok['input_ids'], self.prompt_template_tok['input_ids'])
                    try:
                        assert len(tok['input_ids']) == len(tok['attention_mask'])
                        assert tok['input_ids'][-1] == self.prompt_template_tok['input_ids'][-1]
                        logger.info('fixed it ;)')
                        logger.info(
                            f"new data, token ids={tok['input_ids']}, prompt_token_ids={self.prompt_template_tok['input_ids']}")  # NOQA
                    except AssertionError:
                        logger.info('failed to fix it :( skip it...')


        # toks will be a list of dictionaries, where each dictionary represents the tokenized output of the text from
        # each column in text_columns
        """
            toks = [
                {
                    'input_ids': [token IDs of the first text_column],
                    'attention_mask': [attention mask for the first text_column],
                    # Optional fields like 'token_type_ids' or 'special_tokens_mask' can also be included depending on the tokenizer.
                },
                {
                    'input_ids': [token IDs of the second text_column],
                    'attention_mask': [attention mask for the second text_column],
                    # Optional fields if available.
                },
                {
                    'input_ids': [token IDs of the third text_column],
                    'attention_mask': [attention mask for the third text_column],
                    # Optional fields if available.
                },
                ...
            ]        
        If we have four tokenized columns:
        context with 5 tokens.
        positive with 4 tokens.
        negative with 6 tokens.
        neutral with 4 tokens.
        combined_tok['input_ids'] would be a list of all these tokens concatenated:
        combined_tok['input_ids'] = [101, 102, 103, 104, 105, 201, 202, 203, 204, 301, 302, 303, 304, 305, 306, 303, 304, 305, 306]
        seperate_ids would indicate which tokens come from which column:
        seperate_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
        """
        combined_tok = {}
        seperate_ids = []
        for idx, tok in enumerate(toks):
            for key, val in tok.items():
                if idx == 0:
                    combined_tok[key] = val
                else:
                    combined_tok[key] += val
                if key == 'input_ids':
                    seperate_ids += [idx] * len(val)

        combined_tok['labels'] = [int(data['label']) if 'label' in data else -1]
        combined_tok['seperate_ids'] = seperate_ids
        combined_tok['extra'] = {
            'dataset_format': self.dataset_format,
            'end_with_eos': self.end_with_eos
        }
        return combined_tok


class DichotomyEDataCollator:
    """
    DichotomyEDataCollator. It will be implicitly used in DichotomyE.fit().
    It can only handle the tokenized data using DichotomyEDataTokenizer.

    :param tokenizer:  PreTrainedTokenizerBase
    :param padding:   Union[bool, str, PaddingStrategy], padding strategy
    :param max_length:  Optional[int], max length
    :param return_tensors:  str
    :param filter_duplicate: bool. Whether filter duplicate data
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = 'longest'
    max_length: Optional[int] = None
    return_tensors: str = "pt"
    filter_duplicate: bool = True

    def __init__(self, tokenizer: PreTrainedTokenizerBase, padding: Union[bool, str, PaddingStrategy] = 'longest', max_length: Optional[int] = None, return_tensors: str = "pt", filter_duplicate: bool = True):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.return_tensors = return_tensors
        self. filter_duplicate = filter_duplicate

    def __call__(self, features: List[Dict], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """ Collate function for DichotomyEDataTokenizer.

        :param features: List[Dict]. Tokenized data
        :param return_tensors: str. Default "pt"
        :return: Dict[str, torch.Tensor]. Collated data
        """
        if return_tensors is None:
            return_tensors = self.return_tensors
        has_token_type_ids = "token_type_ids" in features[0]
        end_with_eos = features[0]['extra']['end_with_eos']

        new_features = []
        duplicate_set = set()
        # iterates over each feature in the input list, processes the sequence by splitting based on segmentation
        # markers (seperate_ids), and handles duplicates.
        for feature in features:
            seperate_ids = feature['seperate_ids']
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            assert len(seperate_ids) == len(input_ids) == len(attention_mask)

            # Ensures token_type_ids length matches input_ids, maintaining consistency
            has_token_type_ids = False
            if "token_type_ids" in feature:
                has_token_type_ids = True
                token_type_ids = feature['token_type_ids']
                assert len(token_type_ids) == len(input_ids)

            # The sequence is split into smaller parts based on seperate_ids. This segmentation is crucial for models
            # that need to process parts of sequences independently.
            max_seperate_id = max(seperate_ids)
            prev_start_idx = 0
            current_features = []
            is_duplicate = False
            for seperate_id in range(1, max_seperate_id + 1):
                start_idx = seperate_ids.index(seperate_id)
                new_feature = {}
                new_input_ids = input_ids[prev_start_idx:start_idx]
                if tuple(new_input_ids) in duplicate_set:
                    is_duplicate = True
                    if self.filter_duplicate:
                        break
                duplicate_set.add(tuple(new_input_ids))
                new_feature['input_ids'] = new_input_ids
                new_feature['attention_mask'] = attention_mask[prev_start_idx:start_idx]
                if has_token_type_ids:
                    new_feature['token_type_ids'] = token_type_ids[prev_start_idx:start_idx]
                new_feature['labels'] = feature['labels']
                current_features.append(new_feature)
                prev_start_idx = start_idx

            # last
            new_feature = {}
            new_input_ids = input_ids[prev_start_idx:]
            if tuple(new_input_ids) in duplicate_set:
                is_duplicate = True
            duplicate_set.add(tuple(new_input_ids))
            new_feature['input_ids'] = new_input_ids
            new_feature['attention_mask'] = attention_mask[prev_start_idx:]
            if has_token_type_ids:
                new_feature['token_type_ids'] = token_type_ids[prev_start_idx:]
            new_feature['labels'] = feature['labels']
            current_features.append(new_feature)

            if self.filter_duplicate and is_duplicate:
                continue
            new_features += current_features

        # remove features
        del features

        if end_with_eos:
            features = {}
            features['input_ids'] = [feature['input_ids'] + [self.tokenizer.eos_token_id] for feature in new_features]
            features = self.tokenizer.pad(
                features,
                padding=self.padding,
                return_attention_mask=True,
                return_tensors=return_tensors)
        else:
            features = self.tokenizer.pad(
                {'input_ids': [feature['input_ids'] for feature in new_features]},
                padding=self.padding,
                max_length=self.max_length,
                return_attention_mask=True,
                return_tensors=return_tensors,
            )
        features['labels'] = torch.Tensor([feature['labels'] for feature in new_features])

        return features

