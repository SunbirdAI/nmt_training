import numpy as np 

def postprocess(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_sacreBLEU(eval_preds, eval_languages, samples_per_language, tokenizer, metric, eval_lang = None):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess(decoded_preds, decoded_labels)
    
    result = {}
    for i, lang in enumerate(eval_languages):
        if eval_lang is not None and eval_lang not in lang:
            continue
        result_subset = metric.compute(
            predictions=decoded_preds[i*samples_per_language:(i+1)*samples_per_language],
            references=decoded_labels[i*samples_per_language:(i+1)*samples_per_language])
        result[f"BLEU_{lang}"] = result_subset["score"]
        
    result["BLEU_mean"] = np.mean([result[f"BLEU_{lang}"] for lang in eval_languages])
    
    result = {k: round(v, 4) for k, v in result.items()}
    return result


