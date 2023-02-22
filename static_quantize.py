import datasets 
import transformers

from nmt_clean.config import config
from nmt_clean.load_data import load_training_data, load_validation_data
from nmt_clean.metrics import compute_sacreBLEU as compute_metrics
from nmt_clean.preprocess import Many2OneProcessor

from optimum.intel.neural_compressor import INCQuantizer ,INCSeq2SeqTrainer
from neural_compressor.config import PostTrainingQuantConfig

from transformers import EarlyStoppingCallback


### Quantization not working !!! BLEU scores are not straightforward to optimize for in optimum/neural compressor
config['data_dir'] = f'/home/ali/Documents/repos/datasets/salt/v7-dataset/' #FIXME use os.path.join
config['model_checkpoint'] = "/home/ali/Documents/repos/nmt_checkpoints/mul_en_kaggle_hf_1-2/output-mul-en/checkpoint-400"
tokenizer = transformers.AutoTokenizer.from_pretrained(config["model_checkpoint"])
main_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config['model_checkpoint'])
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model = main_model) 

metric = datasets.load_metric('sacrebleu')

processor = Many2OneProcessor()

#train_dataset = load_training_data(processor, tokenizer)
validation_dataset = load_validation_data(processor, tokenizer)
quantization_config = PostTrainingQuantConfig(approach="static")

quantizer = INCQuantizer.from_pretrained(main_model)

quantizer.quantize(
    quantization_config=quantization_config,
    calibration_dataset=validation_dataset,
    data_collator = data_collator,
    batch_size = 1,
    save_directory="test-quantize-",
)

#os.environ["WANDB_API_KEY"] = secret_value_0
#wandb.init(project=config['wandb_project'],entity=config["wandb_entity"], config=config)

model.config.use_cache = False



# trainer = INCSeq2SeqTrainer(
#     model,
#     config['train_settings'],
#     quantization_config = ,
#     pruning_config = ,
#     distillation_config = ,
#     train_dataset = train_dataset,
#     eval_dataset = validation_dataset,
#     data_collator = data_collator,
#     tokenizer = tokenizer,
#     compute_metrics = lambda x: compute_metrics(
#         x, config['eval_languages'], config['validation_samples_per_language']),
#     callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)],
# )


# trainer.train()