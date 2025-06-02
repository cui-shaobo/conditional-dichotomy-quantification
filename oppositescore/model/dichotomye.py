"""
@Project  : dichotomous-score
@File     : dichotomye.py
@Author   : Shaobo Cui
@Date     : 08.09.2024 12:04
@Reference: https://github.com/SeanLee97/AnglE/

"""

import os
import re
import sys
import json
from functools import partial
from typing import Any, Dict, Optional, List, Union, Tuple

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import Dataset
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM, AutoModel, AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from huggingface_hub import repo_exists

from oppositescore.evaluation.dichotomyInferencer import DichotomyInferencer
from oppositescore.evaluation.evaluation import DichotomyEvaluator
from oppositescore.model.angle import set_device, check_llm, Pooler, find_all_linear_names, DatasetFormats, EvaluateCallback
# from dichotomye_trainer import DichotomyTrainer
from oppositescore.trainer.dichotomy_dataset import DichotomyEDataCollator
from oppositescore.trainer.dichotomye_trainer import DichotomyTrainer
from oppositescore.model.base import AngleBase

from peft import (
    get_peft_model, LoraConfig, TaskType, PeftModel,
    prepare_model_for_kbit_training, PeftConfig,
)

from oppositescore.utils.utils import logger


class DichotomyE(AngleBase):
    """
    DichotomyE: A model similar to AnglE but specifically designed to use DichotomyLoss.

    :param model_name_or_path: str, model name or path.
    :param tokenizer_name_or_path: Optional[str]. Default None. When set to None, it will use the same as `model_name_or_path`.
    :param max_length: int. Default 512
    :param model_kwargs: Optional[Dict]. kwargs for model.
    :param lora_config_kwargs: Optional[Dict]. kwargs for peft lora_config.
    :param pooling_strategy: Optional[str]. Pooling strategy.
    :param apply_lora: Optional[bool]. Whether to apply lora. Default None.
    :param train_mode: bool. Whether to load for training. Default True.
    :param load_kbit: Optional[int]. Specify kbit training from [4, 8, 16]. Default None.
    :param is_llm: Optional[bool]. Whether the model is LLM. Default None.
    :param pretrained_model_path: Optional[str]. Default None.
    :param pretrained_lora_path: Optional[str]. Default None.
    :param torch_dtype: Optional[torch.dtype]. Specify torch_dtype. Default None.
    :param device: Optional[str]. Specify device. Default None.
    :param kbit_kwargs: Optional[Dict]. kwargs for kbit. Default None.
    :param tokenizer_padding_side: Optional[str]. Specify tokenizer padding side from [`left`, `right`]. Default None.
    :param **kwargs: Any other parameters.
    """
    cfg_file_name = 'dichotomy.config'

    def __init__(self,
                 model_name_or_path: str,
                 tokenizer_name_or_path: Optional[str] = None,
                 max_length: int = 512,
                 model_kwargs: Optional[Dict] = None,
                 lora_config_kwargs: Optional[Dict] = None,
                 pooling_strategy: Optional[str] = None,
                 apply_lora: Optional[bool] = None,
                 train_mode: bool = True,
                 load_kbit: Optional[int] = None,
                 is_llm: Optional[bool] = None,
                 pretrained_model_path: Optional[str] = None,
                 pretrained_lora_path: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 device: Optional[str] = None,
                 kbit_kwargs: Optional[Dict] = None,
                 tokenizer_padding_side: Optional[str] = None,
                 apply_billm: bool = False,
                 billm_model_class: Optional[str] = None,
                 cache_hf_dir: Optional[str] = None,
                 **kwargs: Any):
        super().__init__()
        self.max_length = max_length
        self.train_mode = train_mode
        self.pooling_strategy = pooling_strategy
        self.load_kbit = load_kbit
        self.is_llm = is_llm
        self.cache_hf_dir = cache_hf_dir
        if device:
            self.device = device
        else:
            self.device = set_device()
        if is_llm is None:
            self.is_llm = check_llm(model_name_or_path)
            if self.is_llm:
                logger.info('LLM detected, automatically set is_llm=True.'
                            'If it is wrong, you can manually set `is_llm`.')
        if self.is_llm and self.pooling_strategy != 'last':
            logger.info(f'🚨 LLM detected, but pooling strategy is specified to {self.pooling_strategy}.'
                        'Please check whether it is correct. It is recommended to use `last` pooling strategy for LLM.')

        self.apply_lora = apply_lora
        if self.apply_lora is None:
            if self.is_llm:
                self.apply_lora = True
                logger.info('LLM detected, automatically set apply_lora=True.'
                            'If it is wrong, you can manually set `apply_lora`.')
            if pretrained_lora_path is not None:
                self.apply_lora = True

        if self.device == 'cuda':
            self.gpu_count = torch.cuda.device_count()
        elif self.device == 'mps':
            self.gpu_count = 1
        else:
            self.gpu_count = 0

        if torch_dtype is None:
            torch_dtype = torch.float32 if train_mode else None

        lora_config = None
        if self.apply_lora:
            lora_config = {
                'task_type': TaskType.FEATURE_EXTRACTION,
                'r': 32,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
            }
            if lora_config_kwargs is not None:
                lora_config.update(lora_config_kwargs)
            if train_mode:
                logger.info(f'lora_config={lora_config}')

        # Tokenizer
        # This is emergent changes from AoE model.
        if self.is_llm:
            # config = PeftConfig.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path or model_name_or_path, trust_remote_code=True, cache_dir=self.cache_hf_dir)
        if tokenizer_padding_side is not None and self.tokenizer.padding_side != tokenizer_padding_side:
            self.tokenizer.padding_side = tokenizer_padding_side
        if self.is_llm and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        # Backbone model
        model_kwargs = model_kwargs if model_kwargs is not None else {}
        kbit_kwargs = kbit_kwargs if kbit_kwargs is not None else {}
        if self.is_llm:
            device_map = "auto"
            if apply_billm:
                assert billm_model_class is not None, "billm_model_class should be specified for apply_billm=True"
                try:
                    import billm
                except ImportError as err:
                    print(f'Import Error: {err}')
                    print('Please install the latest billm via: python -m pip install -U billm')
                    raise

                MODEL_CLASS = getattr(billm, billm_model_class)
            else:
                MODEL_CLASS = AutoModelForCausalLM
            if train_mode and self.gpu_count > 1:
                device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            # LLM
            if self.apply_lora:
                lora_config['bias'] = "none"
                lora_config['task_type'] = TaskType.CAUSAL_LM

                is_kbit = load_kbit in [4, 8]
                if is_kbit:
                    model = MODEL_CLASS.from_pretrained(
                        model_name_or_path,
                        config=None,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=load_kbit == 4,
                            load_in_8bit=load_kbit == 8,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                            bnb_4bit_compute_dtype=torch.float32,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type='nf4',
                        ),
                        torch_dtype=torch.float32,
                        device_map=device_map,
                        trust_remote_code=True,
                        cache_dir = self.cache_hf_dir,
                    )
                else:
                    model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                        device_map=device_map,
                                                        output_hidden_states=True,
                                                        trust_remote_code=True,
                                                        torch_dtype=torch_dtype or torch.float16,
                                                        cache_dir=self.cache_hf_dir)
                if train_mode and is_kbit:
                    model = prepare_model_for_kbit_training(model, **kbit_kwargs)

                if pretrained_lora_path is not None:
                    logger.info(f'Load lora weight from {pretrained_lora_path}')
                    model = PeftModel.from_pretrained(
                        model,
                        pretrained_lora_path,
                        torch_dtype=torch.float32 if is_kbit else (torch_dtype or torch.float16),
                        device_map=device_map,
                        is_trainable=train_mode,
                        cache_dir=self.cache_hf_dir
                    )
                elif train_mode:
                    if 'target_modules' not in lora_config or lora_config.get('target_modules', None) is None:
                        target_modules = find_all_linear_names(
                            model, linear_type=bnb.nn.Linear4bit if load_kbit == 4 else nn.Linear)
                        lora_config['target_modules'] = target_modules
                        logger.info(f'lora target modules={target_modules}')
                    peft_config = LoraConfig(**lora_config)
                    model = get_peft_model(model, peft_config)

                if is_kbit:
                    model = DichotomyE.kbit_post_handle(model)

                self.backbone = model
            else:
                model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                    device_map=device_map,
                                                    output_hidden_states=True,
                                                    trust_remote_code=True,
                                                    torch_dtype=torch_dtype or torch.float16,
                                                    cache_dir=self.cache_hf_dir)
                self.backbone = model
        else:
            # non-LLMs
            if self.apply_lora:
                model = AutoModel.from_pretrained(pretrained_model_path or model_name_or_path, trust_remote_code=True, cache_dir=self.cache_hf_dir)
                if pretrained_lora_path is not None:
                    model = PeftModel.from_pretrained(
                        model,
                        pretrained_lora_path,
                        is_trainable=train_mode
                    )
                else:
                    if 'target_modules' not in lora_config or lora_config.get('target_modules', None) is None:
                        target_modules = find_all_linear_names(model)
                        lora_config['target_modules'] = target_modules
                        logger.info(f'lora target modules={target_modules}')
                    peft_config = LoraConfig(**lora_config)
                    model = get_peft_model(model, peft_config)
                self.backbone = model
            else:
                if pretrained_model_path is not None:
                    logger.info(f'Load pretrained model from {pretrained_model_path}')
                self.backbone = AutoModel.from_pretrained(
                    pretrained_model_path or model_name_or_path,
                    trust_remote_code=True,
                    cache_dir=self.cache_hf_dir)

        if train_mode and self.apply_lora:
            self.backbone.print_trainable_parameters()

        self.backbone.config.use_cache = False

        # Pooling strategy
        self.pooler = Pooler(
            self.backbone,
            pooling_strategy=self.pooling_strategy,
            padding_strategy=self.tokenizer.padding_side)

        # For configuration
        self.__cfg = {
            'model_name_or_path': model_name_or_path,
            'max_length': max_length,
            'model_kwargs': model_kwargs,
            'pooling_strategy': pooling_strategy,
            'lora_config_kwargs': lora_config,
            'is_llm': self.is_llm,
            'apply_billm': apply_billm,
            'billm_model_class': billm_model_class,
            'apply_lora': self.apply_lora,
            'tokenizer_padding_side': tokenizer_padding_side,
        }
        self.__cfg.update(kwargs)

    def cuda(self):
        if self.gpu_count > 1:
            return self
        if self.gpu_count > 1 and self.is_llm:
            return self
        else:
            self.backbone = self.backbone.to(torch.device(self.device))
            return self

    def to(self, device: Any):
        if isinstance(device, str):
            device = torch.device(device)
        self.backbone = self.backbone.to(device)
        self.device = device
        return self

    @staticmethod
    def kbit_post_handle(model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.float32)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    module = module.to(torch.float32)
        return model

    @staticmethod
    def find_pth_path(dirpath: str, config: Dict) -> str:
        if config['save_mode'] == 'best':
            return os.path.join(dirpath, config['best_file_name'])

        pth_list = []
        for fname in os.listdir(dirpath):
            if fname.endswith('.pth'):
                epoch = int(re.search(r'\d+', fname).group())
                pth_list.append((epoch, fname))
        pth_list = sorted(pth_list, key=lambda x: x[0], reverse=True)
        return os.path.join(dirpath, pth_list[0][1])

    @staticmethod
    def from_pretrained(model_name_or_path: str,
                        cached_hf_dir: Optional[str] = None,
                        pretrained_model_path: Optional[str] = None,
                        pretrained_lora_path: Optional[str] = None,
                        is_llm: Optional[bool] = None,
                        pooling_strategy: str = 'cls',
                        train_mode: bool = False,
                        **kwargs):
        """
        Load AnglE from pretrained model.

        :param model_name_or_path: str, model name or path. Required.
        :param pretrained_model_path: Optional[str].
        :param pretrained_lora_path: Optional[str].
        :param is_llm: Optional[bool].
        :param pooling_strategy: str. Pooling Strategy. Default `cls`.
        :param train_mode: bool. Default False.
        :param kwargs: Other kwargs for AnglE.

        :return: DichotomyE object.

        Example::

                from angle_emb import AnglE

                angle = AnglE.from_pretrained(model_name_or_path)
                # fit
                angle.fit(*args, **kwargs)
                # inference
                angle.encode(*args, **kwargs)
        """
        dichotomye = DichotomyE(model_name_or_path,
                      is_llm=is_llm,
                      pretrained_model_path=pretrained_model_path,
                      pretrained_lora_path=pretrained_lora_path,
                      pooling_strategy=pooling_strategy,
                      train_mode=train_mode,
                      cache_hf_dir=cached_hf_dir,
                      **kwargs)
        return dichotomye

    @staticmethod
    def load_config(fpath: str) -> Dict:
        with open(fpath, 'r', encoding='utf-8') as reader:
            return json.load(reader)

    def save_config(self, fpath: str):
        with open(fpath, 'w', encoding='utf-8') as writer:
            json.dump(self.__cfg, writer, ensure_ascii=False, indent=2)

    def detect_dataset_format(self, ds: Dataset):
        for obj in ds:
            return obj['extra']['dataset_format']


    def fit(self,
            train_ds: Dataset,
            valid_ds: Optional[Dataset] = None,
            batch_size: int = 32,
            output_dir: Optional[str] = None,
            epochs: int = 1,
            learning_rate: float = 1e-5,
            warmup_steps: int = 1000,
            logging_steps: int = 10,
            eval_steps: Optional[int] = None,
            save_steps: int = 100,
            save_strategy: str = 'steps',
            save_total_limit: int = 10,
            gradient_accumulation_steps: int = 1,
            fp16: Optional[bool] = None,
            argument_kwargs: Optional[Dict] = None,
            trainer_kwargs: Optional[Dict] = None,
            loss_kwargs: Optional[Dict] = None,
            apply_ese: bool = False,
            filter_duplicate: bool = True,
            push_to_hub: bool = False,
            hub_model_id: Optional[str] = None,
            hub_private_repo: bool = True):
        """
        Fit using DichotomyE.

        :param train_ds: Dataset. tokenized train dataset. Required.
        :param valid_ds: Optional[Dataset]. tokenized valid dataset. Default None.
        :param batch_size: int. Default 32.
        :param output_dir: Optional[str]. save dir. Default None.
        :param epochs: int. Default 1.
        :param learning_rate: float. Default 1e-5.
        :param warmup_steps: int. Default 1000.
        :param logging_steps: int. Default 10.
        :param eval_steps: Optional[int]. Default None.
        :param save_steps: int. Default 100.
        :param save_strategy: str. Default steps.
        :param save_total_limit: int. Default 10.
        :param gradient_accumulation_steps: int. Default 1.
        :param fp16: Optional[bool]. Default None.
        :param argument_kwargs: Optional[Dict]. kwargs for TrainingArguments.
            refer to: https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/trainer#transformers.TrainingArguments
        :param trainer_kwargs: Optional[Dict]. kwargs for AngleTrainer.
        :param loss_kwargs: Optional[Dict]. kwargs for AngleLoss.
        :param apply_ese: bool, whether apply ESE training.
        :param filter_duplicate: bool, whether filter duplicate samples.
        :param push_to_hub: bool, whether push to hub.
        :param hub_model_id: Optional[str], hub model id.
        :param hub_private_repo: bool, whether push to private repo.
        """  # NOQA
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # save config
        self.save_config(os.path.join(output_dir, DichotomyE.cfg_file_name))
        # save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        if self.gpu_count > 1:
            gradient_accumulation_steps = gradient_accumulation_steps // self.gpu_count
        if fp16 is None and self.is_llm:
            fp16 = True
        else:
            fp16 = False

        # init argument_kwargs
        if argument_kwargs is None:
            argument_kwargs = {}
        if 'push_to_hub' not in argument_kwargs:
            argument_kwargs['push_to_hub'] = push_to_hub
        if 'hub_model_id' not in argument_kwargs:
            argument_kwargs['hub_model_id'] = hub_model_id
        if 'hub_private_repo' not in argument_kwargs:
            argument_kwargs['hub_private_repo'] = hub_private_repo

        if trainer_kwargs is None:
            trainer_kwargs = {}

        callbacks = None

        # Now it only supports for DatasetFormats.A
        if valid_ds is not None:
            # check format
            for obj in valid_ds:
                if obj['extra']['dataset_format'] != DatasetFormats.A:
                    raise ValueError('Currently only support evaluation for DatasetFormats.A.')
                break
            best_ckpt_dir = None
            if output_dir is not None:
                best_ckpt_dir = os.path.join(output_dir, 'best-checkpoint')
            evaluate_callback = EvaluateCallback(self, valid_ds,
                                                 partial(self.evaluate, batch_size=batch_size),
                                                 save_dir=best_ckpt_dir,
                                                 push_to_hub=push_to_hub,
                                                 hub_model_id=hub_model_id,
                                                 hub_private_repo=hub_private_repo)
            # set False to ensure only best checkpoint is pushed
            argument_kwargs['push_to_hub'] = False
            callbacks = [evaluate_callback]

        CustomTrainer = DichotomyTrainer if apply_ese else DichotomyTrainer
        trainer = CustomTrainer(
            pooler=self.pooler,
            model=self.backbone,
            dataset_format=self.detect_dataset_format(train_ds),
            train_dataset=train_ds,
            eval_dataset=None,
            loss_kwargs=loss_kwargs,
            tokenizer=self.tokenizer,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                fp16=fp16,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                eval_steps=eval_steps,
                save_steps=save_steps,
                output_dir=output_dir,
                save_total_limit=save_total_limit,
                load_best_model_at_end=False,
                ddp_find_unused_parameters=False if self.gpu_count > 1 else None,
                label_names=['labels', 'seperate_ids', 'extra'],
                **argument_kwargs,
            ),
            callbacks=callbacks,
            data_collator=DichotomyEDataCollator(
                self.tokenizer, return_tensors="pt", max_length=self.max_length, filter_duplicate=filter_duplicate
            ),
            **trainer_kwargs
        )
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.backbone = torch.compile(self.backbone)

        trainer.train()
        if argument_kwargs.get('push_to_hub', False):
            trainer.push_to_hub()
        self.backbone.save_pretrained(output_dir)

    # def evaluate(self, data: Dataset, batch_size: int = 32, metric: str = 'spearman_cosine') -> float:
    #     """ evaluate
    #
    #     :param data: Dataset, DatasetFormats.A is required
    #     :param batch_size: int. Default 32.
    #     :param metric: str. Default 'spearman_cosine'.
    #
    #     :return: float.
    #     """
    #     return CorrelationEvaluator(
    #         text1=data['text1'],
    #         text2=data['text2'],
    #         labels=data['label'],
    #         batch_size=batch_size,
    #     )(self)[metric]
    def evaluate(self, data: Dataset, batch_size: int = 32) -> float:
        """ Evaluate using DichotomyEvaluator.

        :param data: Dataset, must include columns `context`, `positive`, `negative`, and `neutral`.
        :param batch_size: int. Default 32.

        :return: dict with DCF metric.
        """
        evaluator = DichotomyEvaluator(
            context=data['context'],
            positive=data['positive'],
            negative=data['negative'],
            neutral=data['neutral'],
            batch_size=batch_size
        )
        return evaluator(self)

    def calculate_opposite_score(self, ctx, sent1, sent2, batch_size: int = 1):
        inferencer = DichotomyInferencer(
            context=ctx,
            sent1=sent1,
            sent2=sent2,
            batch_size=batch_size
        )

        return inferencer(self)


    def truncate_layer(self, layer_index: int):
        """ truncate layer

        :param layer_index: int. layers after layer_index will be truncated.
        :return: self
        """
        if len(self.backbone.encoder.layer) < layer_index:
            logger.info('current layer_index is larger than the number of layers, please check whether it is correct')
        self.backbone.encoder.layer = self.backbone.encoder.layer[:layer_index]
        return self

    def encode(self,
               inputs: Union[List[str], Tuple[str], List[Dict], str],
               max_length: Optional[int] = None,
               end_with_eos: bool = False,
               to_numpy: bool = True,
               embedding_start: int = 0,
               embedding_size: Optional[int] = None,
               device: Optional[Any] = None,
               prompt: Optional[str] = None,
               normalize_embedding: bool = False):
        """
        encode texts.

        :param inputs: Union[List[str], Tuple[str], List[Dict], str]. Input texts. Required.
        :param max_length: Optional[int]. Default None.
        :param to_numpy: bool. Default True.
        :param embedding_start: int. Specify the start position of the embedding (for Espresso).
        :param embedding_size: Optional[int]. Specify embedding size (for Espresso).
            The embeddings from embedding_start to embedding_start+embedding_size will be returned.
        :param device: Optional[Any]. Default None.
        :param prompt: Optional[str]. Default None.
        :param normalize_embedding: bool. Default False.
        """
        self.backbone.eval()

        if device is None:
            device = self.device
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        if prompt is not None:
            for i, obj in enumerate(inputs):
                assert isinstance(obj, dict), 'The prompt has been set, please pass a dict like {"prompt_key": "text"}'
                inputs[i] = prompt.format(**obj)
        max_length = max_length or self.max_length
        if end_with_eos:
            max_length -= 1

        if end_with_eos:
            tok = self.tokenizer(
                inputs,
                padding=False,
                return_attention_mask=False,
                max_length=max_length or self.max_length,
                truncation=True)
            tok['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in tok['input_ids']]
            tok = self.tokenizer.pad(tok, padding=True, return_attention_mask=True, return_tensors='pt')
        else:
            tok = self.tokenizer(
                inputs,
                padding='longest',
                max_length=max_length or self.max_length,
                truncation=True,
                return_tensors='pt')
        tok.to(device)
        with torch.no_grad():
            output = self.pooler(tok,
                                 embedding_start=embedding_start,
                                 embedding_size=embedding_size)
        if normalize_embedding:
            output = nn.functional.normalize(output, p=2, dim=-1)
        if to_numpy:
            return output.float().detach().cpu().numpy()
        return output

    def push_to_hub(self, hub_model_id: str, private: bool = True, exist_ok: bool = False, **kwargs):
        """ push model to hub

        :param hub_model_id: str, hub model id.
        :param private: bool, whether push to private repo. Default True.
        :param exist_ok: bool, whether allow overwrite. Default False.
        :param kwargs: other kwargs for `push_to_hub` method.
        """
        if not exist_ok and repo_exists(hub_model_id):
            raise ValueError(f"Model {hub_model_id} already exists on the hub. Set `exist_ok=True` to overwrite.")
        self.tokenizer.push_to_hub(hub_model_id, private=private, **kwargs)
        self.backbone.push_to_hub(hub_model_id, private=private, **kwargs)

    def save_pretrained(self, output_dir: str, exist_ok: bool = True):
        """ save model and tokenizer

        :param output_dir: str, output dir.
        :param exist_ok: bool, whether allow overwrite. Default True.
        """
        if not exist_ok and os.path.exists(output_dir):
            raise ValueError(f"Output directory ({output_dir}) already exists and is not empty.")
        os.makedirs(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.backbone.save_pretrained(output_dir)

