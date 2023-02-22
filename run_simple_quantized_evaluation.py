
import datasets
import evaluate
import transformers

from nmt_clean.config import config
from nmt_clean.evaluator import translate_one
from nmt_clean.load_data import load_testing_data
from nmt_clean.metrics import compute_sacreBLEU as compute_metrics
from nmt_clean.preprocess import Many2OneProcessor

from optimum.intel.neural_compressor import  IncQuantizedModelForSeq2SeqLM, INCSeq2SeqTrainer
from transformers import Seq2SeqTrainer



config['data_dir'] = f'/home/ali/Documents/repos/datasets/salt/v7-dataset/' #FIXME use os.path.join
config['model_checkpoint'] = "/home/ali/Documents/repos/NMT_clean/output-salt-en"
#config['tokenizer_checkpoint'] = "/home/ali/Documents/repos/nmt_checkpoints/mul_en_kaggle_hf_1-2/output-mul-en/checkpoint-400"


#main_model = INCModelForSequenceClassification.from_pretrained(config['model_checkpoint'])
main_model =  IncQuantizedModelForSeq2SeqLM.from_pretrained(config['model_checkpoint'])
tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_checkpoint'])
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model = main_model) 

#bleu = evaluate.load("bleu")
sacrebleu = datasets.load_metric('sacrebleu')

processor = Many2OneProcessor()
test_dataset = load_testing_data(processor, tokenizer)

trainer = INCSeq2SeqTrainer(
    main_model,
    config['train_settings'],
    train_dataset = None,
    eval_dataset = test_dataset,
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = lambda x: compute_metrics(
        x, config['eval_languages'], config['validation_samples_per_language'],tokenizer,sacrebleu)
)

trainer.config = config['train_settings']
trainer.config.max_length = config["max_input_length"] #FIXME issue PR to the repo
trainer.config.num_beams = 5 #FIXME issue PR to the repo


results = trainer.evaluate()
print(results)