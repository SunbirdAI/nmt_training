
# nmt_training
A repo to contain non-evolutionary NMT experiments (might merge with evobird later)
The main idea is to hide the necessary data-preprocessing and loading functions behind preprocess classes. Allowing training to proceed smoothly whether with huggingface trainer objects, pytorch training loops, evotorch searchers or something else.

## How to load Many2One data
1) Modify the necessary configurations in the config.py file
```
config["data_dir"] = /my/own/path/to/data/dir/v7-dataset/
```

2) Load the data using the Many2OneProcessor class
```
from  nmt_clean.load_data  import  load_training_data, load_validation_data, load_testing_data
from  nmt_clean.preprocess  import  Many2OneProcessor


processor = Many2OneProcessor()
train_dataset = load_training_data(processor, tokenizer)
validation_dataset = load_validation_data(processor, tokenizer)
test_dataset = load_testing_data(processor, tokenizer)
```
Now you have your tokenized data ready to feed into a model

## How to load Many2Many data
1) Modify the necessary configurations in the config_m2m_mul_en.py file

```
config["data_dir"] = /my/own/path/to/data/dir/v7-dataset/
```

2) Load the data using the Many2OneProcessor class
```
from  nmt_clean.load_data  import  load_training_data, load_validation_data, load_testing_data
from  nmt_clean.preprocess  import  Many2ManyProcessor

preprocess = Many2ManyProcessor()

paired_datasets = preprocess.dataset_from_folders_m2m(config["training_subset_paths"])
tokens = preprocess.iterative_preprocess(paired_datasets,tokenizer )```
