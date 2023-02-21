
import datasets
import evaluate
import transformers

from nmt_clean.evaluate import translate_one
from nmt_clean.preprocess import Many2OneProcessor
from transformers import Seq2SeqTrainer


config['data_dir'] = f'/home/ali/Documents/repos/datasets/salt/v7-dataset/' #FIXME use os.path.join
config['model_checkpoint'] = "/home/ali/Documents/repos/nmt_checkpoints/mul_en_kaggle_hf_1-2/output-mul-en/checkpoint-400"
tokenizer = transformers.AutoTokenizer.from_pretrained(config["model_checkpoint"])
main_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config['model_checkpoint'])
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model = main_model) 

#bleu = evaluate.load("bleu")
sacrebleu = datasets.load_metric('sacrebleu')

processor = M21RawTextProcessor()
test_dataset = load_validation_data(processor, tokenizer)

trainer = Seq2SeqTrainer(
    model,
    config['train_settings'],
    train_dataset = None,
    eval_dataset = test_dataset,
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = lambda x: compute_metrics(
        x, config['eval_languages'], config['validation_samples_per_language']),
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)],
)


results = trainer.evaluate()
print(results)
