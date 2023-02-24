import datasets
import numpy as np

from nmt_clean.config import config 


## TODO either class these functions or combine them into 1 function
def load_training_data(processor, tokenizer,
training_subset_paths = config['training_subset_paths'],
token_conversion_dict = config["token_conversion_dict"],
validation_train_merge = config["validation_train_merge"],
validation_subset_paths = config["validation_subset_paths"],
validation_samples_per_language = config["validation_samples_per_language"]):
    training_subsets = processor.load_datasets(training_subset_paths,
                                            token_conversion_dict,
                                            validation_cutoff = 0)
    if config["validation_train_merge"]: #WARNING: this will introduce an indirect leak because the validation set will be a temporary test set as well
        extra_training_data = processor.load_datasets(validation_subset_paths, 
                                                token_conversion_dict,
                                                validation_cutoff = validation_samples_per_language,
                                                mode = "cutoff_minimum")
        training_subsets.extend(extra_training_data)

    training_subsets = [s.shuffle() for s in training_subsets]


    sample_probabilities = np.array([len(s) for s in training_subsets])
    #sample_probabilities[6:9] = sample_probabilities[6]//10 #downsample mt560 ach by a factor of 10
    #sample_probabilities[6:9] = sample_probabilities[6]//20 #downsample mt560 lug by a factor of 20
    #sample_probabilities[6:9] = sample_probabilities[6]//10 #downsample mt560 nyn by a factor of 10

    sample_probabilities = sample_probabilities / np.sum(sample_probabilities)

    train_data_raw = datasets.interleave_datasets( 
        training_subsets, sample_probabilities)

    train_data  = train_data_raw.map(
        lambda x:processor.preprocess(x, tokenizer), remove_columns=["translation"], batched=True)


    return train_data

def load_validation_data(processor, tokenizer,
validation_subset_paths = config["validation_subset_paths"],
token_conversion_dict = config["token_conversion_dict"],
validation_samples_per_language= config['validation_samples_per_language']):

    validation_subsets = processor.load_datasets(validation_subset_paths, 
                                                    token_conversion_dict,
                                                validation_cutoff = validation_samples_per_language,
                                                mode = "cutoff_maximum")
        

    validation_data_raw = datasets.concatenate_datasets(validation_subsets)


    validation_data  = validation_data_raw.map(
        lambda x:processor.preprocess(x, tokenizer), remove_columns=["translation"], batched=True)

    return validation_data

def load_testing_data(processor, tokenizer,
testing_subset_paths = config['testing_subset_paths'],
token_conversion_dict = config["token_conversion_dict"],
testing_samples_per_language = config['testing_samples_per_language']):

    testing_subsets = processor.load_datasets(testing_subset_paths, 
                                                    token_conversion_dict,
                                                validation_cutoff = testing_samples_per_language,
                                                mode = "cutoff_maximum")
        

    testing_data_raw = datasets.concatenate_datasets(testing_subsets)


    test_data  = testing_data_raw.map(
        lambda x:processor.preprocess(x, tokenizer), remove_columns=["translation"], batched=True)

    return test_data


#FIXME processor class
def load_raw_text(data= "test", scarebleu = True,
train_data_dir = config["data_dir"],
#training_samples_per_language = config['training_samples_per_language'],
validation_data_dir = config["data_dir"],
validation_samples_per_language = config['validation_samples_per_language'],
test_data_dir = config["data_dir"],
test_samples_per_language = config['testing_samples_per_language'],
eval_languages = config['eval_languages']):

    sources = {}
    references = {}
    sacrereferences = {}
    
    if data == "train":
        data_dir = train_data_dir
        N = 0

    elif data == "validation":
        data_dir = validation_data_dir
        N = validation_samples_per_language

    elif data == "test":
        data_dir = test_data_dir
        N = test_samples_per_language

    for language in eval_languages:
        sources[language] = _file_to_list(
            os.path.join(data_dir + f'test_{language}.src') )
        sources[language] = [source_language_token[language] + ' ' + s
                            for s in sources[language]]
        sources[language] = sources[language][:N]
        references[language] = _file_to_list(
             os.path.join(data_dir + f'test_{language}.tgt') )
        references[language] = [source_language_token['en'] + ' ' + s
                                for s in references[language]]
        references[language] = references[language][:N]
        
        if sacrebleu:
            sacrereferences[language] = [ [source_language_token['en'] + ' ' + s]
                                    for s in references[language]]
            
            sacrereferences[language] = sacrereferences[language][:N]
            return sources, references, sacrereferences
        
        return sources, references
