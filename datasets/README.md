
## Irrelevant Sentence Generation

1\. Make a new file.
```bash
touch api_key/config.json
```
2\. Add the OpenAI API key.

```json
{
    "openai_api_key": "..."
}
```


3\. Run `dichotomy_irrelevant.py`. 

```bash
python neutral_generation_irrelevant_various_pattern_main.py --data_dir 
```

Arguments:
```
usage: dichotomy_irrelevant.py [-h] [--data_dir DATA_DIR]
                               [--dataset_name DATASET_NAME] [--batch_size BATCH_SIZE]
                               [--split SPLIT] [--start START] [--gpt_model GPT_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR
  --dataset_name DATASET_NAME
  --batch_size BATCH_SIZE
  --split SPLIT
  --start START
  --gpt_model GPT_MODEL
```

5\. The processed data are saved in `data/{dataset_name}` folder with `test_processed_{model_name}.jsonl`.

