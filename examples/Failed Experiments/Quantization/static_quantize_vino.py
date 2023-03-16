import datasets 
import torch 
import transformers

from nmt_clean.config import config
from nmt_clean.load_data import load_training_data, load_validation_data
from nmt_clean.metrics import compute_sacreBLEU as compute_metrics
from nmt_clean.preprocess import Many2OneProcessor


from optimum.intel.neural_compressor import INCQuantizer ,INCSeq2SeqTrainer
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig

from transformers import EarlyStoppingCallback
from transformers import Seq2SeqTrainer, EarlyStoppingCallback

from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Callable, ClassVar, Dict, Optional, Union

### Quantization not working !!! BLEU scores are not straightforward to optimize for in optimum/neural compressor
config["validation_samples_per_language"] = 10
config['data_dir'] = f'/home/ali/Documents/repos/datasets/salt/v7-dataset/' #FIXME use os.path.join
config['model_checkpoint'] = "/home/ali/Documents/repos/nmt_checkpoints/mul_en_kaggle_hf_1-2/output-mul-en/checkpoint-400"
tokenizer = transformers.AutoTokenizer.from_pretrained(config["model_checkpoint"])
main_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config['model_checkpoint'])
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model = main_model) 

metric = datasets.load_metric('sacrebleu')

processor = Many2OneProcessor()

#train_dataset = load_training_data(processor, tokenizer)
validation_dataset = load_validation_data(processor, tokenizer,validation_samples_per_language = config["validation_samples_per_language"])
quantization_config = PostTrainingQuantConfig(approach="static")

val_dataloader = torch.utils.data.DataLoader(
                     validation_dataset,
                     batch_size=24, shuffle=False,
                     num_workers=1)
trainer = Seq2SeqTrainer(
    main_model,
    config['train_settings'],
    train_dataset = None,
    eval_dataset = validation_dataset,
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = lambda x: compute_metrics(
        x, config['eval_languages'], config['validation_samples_per_language'],
        tokenizer, metric),
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)],
    )

def quant_calib_fn(model, *args, **kwargs):
    
    results = trainer.evaluate()
    return results["eval_BLEU_mean"]




# Quantization code

conf = PostTrainingQuantConfig()
q_model = quantization.fit(model=main_model,
                           conf=conf,
                           calib_dataloader=val_dataloader,
                           calib_func= quant_calib_fn)
trainer.model = q_model
trainer.evaluate()
q_model.save('./output')


