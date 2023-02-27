
import datasets
import evaluate

from nmt_clean.evaluator import translate_one
from nmt_clean.preprocess import M21RawTextProcessor

import transformers

mul_en_checkpoint_path = "/content/mBART_weights/best_mBART_salt"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
#bleu = evaluate.load("bleu")
sacrebleu = datasets.load_metric('sacrebleu')

processor = M21RawTextProcessor()
sources, references, sacrereferences = processor.preprocess(config["testing_subset_paths"], config["testing_samples_per_language"])

predictions_labels_here = []

predictions = {}
#ach lgg lug nyn teo config['eval_languages']
eval_iter = iter(config['eval_languages'])
for language in config['eval_languages']:
    print(language)

    predictions[language] = []
    counter = 0
    for text in tqdm.tqdm(sources[language] ):
        predictions[language].append(
            translate_one(text, mul_en_model, mul_en_tokenizer))


results = {}
results_sacre = {}
for language in config['eval_languages']: 
    #results[language] = bleu.compute(
    #    predictions=predictions[language][:N],
    #   references=references[language][:N])
    results_sacre[language] = sacrebleu.compute(
         predictions=predictions[language][:N],
         references=sacrereferences[language][:N])
    
    #print(f"BLEU_{language} = {100 * results[language]['bleu']:.1f}")
    print(f"sacreBLEU_{language} = {results_sacre[language]['score']:.1f}")