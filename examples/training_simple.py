import datasets 
import transformers

from nmt_clean.config import config
from nmt_clean.load_data import load_training_data, load_validation_data, load_testing_data
from nmt_clean.metrics import compute_sacreBLEU as compute_metrics
from nmt_clean.preprocess import Many2OneProcessor

from transformers import EarlyStoppingCallback


config['data_dir'] = f'/home/ali/Documents/repos/datasets/salt/v7-dataset/' #FIXME use os.path.join
config['model_checkpoint'] = "/home/ali/Documents/repos/nmt_checkpoints/mul_en_kaggle_hf_1-2/output-mul-en/checkpoint-400"
tokenizer = transformers.AutoTokenizer.from_pretrained(config["model_checkpoint"])
main_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config['model_checkpoint'])
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model = main_model) 

if not config["recycle_language_tokens"]:
    offset = tokenizer.sp_model_size + tokenizer.fairseq_offset
    tokenizer.lang_code_to_id = {}
    for i, code in enumerate(config["SALT_LANGUAGE_CODES"]):
        tokenizer.lang_code_to_id[code] = i + offset
        tokenizer.fairseq_ids_to_tokens[i + offset] = code

metric = datasets.load_metric('sacrebleu')

processor = Many2OneProcessor()

#train_dataset = load_training_data(processor, tokenizer)
validation_dataset = load_validation_data(processor, tokenizer)
test_dataset = load_testing_data(processor, tokenizer)


#os.environ["WANDB_API_KEY"] = secret_value_0
#wandb.init(project=config['wandb_project'],entity=config["wandb_entity"], config=config)

main_model.config.use_cache = False

trainer = transformers.Seq2SeqTrainer(
    main_model,
    config['train_settings'],
    train_dataset = validation_dataset,
    eval_dataset = test_dataset,
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = lambda x: compute_metrics(
        x, config['eval_languages'], config['validation_samples_per_language'], tokenizer,metric ),
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)],
)
#trainer.config = config['train_settings']
#trainer.config.max_length = config["max_input_length"] #FIXME issue PR to the repo
#trainer.config.num_beams = 5 #FIXME issue PR to the repo

train_result = trainer.train()
metrics = trainer.evaluate()
print(metrics)
trainer.save_model()