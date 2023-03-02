import datasets
import sacremoses

from nmt_clean.config import config 
from tqdm import tqdm

def sentence_format(input):
    if input == "":
        print("empty string")
        return ""
    '''Ensure capital letter at the start and full stop at the end.'''
    input = input[0].capitalize() + input[1:]
    if input[-1] not in ['.', '!', '?']:
        input = input + '.'

    return input
    

class Processor():

    @staticmethod
    def load_files(path_to_file):
        with open(path_to_file) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            return lines


    def load_datasets(self, pair_dicts_list, language_token_dict, validation_cutoff = 0,mode = "cutoff_maximum"):
        list_of_pairs = []
        for language_pair_dict in pair_dicts_list:
            src_language = language_pair_dict["source"]["language"]
            src_scentences = self.load_files(language_pair_dict["source"]["path"])
            tgt_language = language_pair_dict["target"]["language"]
            tgt_scentences = self.load_files(language_pair_dict["target"]["path"])
                    
            if validation_cutoff:
                if mode == "cutoff_maximum":
                    src_scentences = src_scentences[:validation_cutoff]
                    tgt_scentences = tgt_scentences[:validation_cutoff]
                elif mode == "cutoff_minimum":
                    src_scentences = src_scentences[validation_cutoff:]
                    tgt_scentences = tgt_scentences[validation_cutoff:]

            
            normalizer = sacremoses.MosesPunctNormalizer()

            src_scentences = [sentence_format(normalizer.normalize(text))
                    for text in src_scentences]
            tgt_scentences = [sentence_format(normalizer.normalize(text))
                    for text in tgt_scentences]
            
            # No src language token here            

            pairs = {'translation': [{"src": s,
                                "tgt": t}
                                for s, t in zip(src_scentences, tgt_scentences)]}

            list_of_pairs.append(datasets.Dataset.from_dict(pairs))

        return list_of_pairs

    @staticmethod
    def preprocess(examples, tokenizer,
    max_input_length = config['max_input_length'],
    max_target_length = config['max_target_length'],
    teacher_forcing = True ):
        normalizer = sacremoses.MosesPunctNormalizer()
        inputs = [ex["src"] for ex in examples['translation']]
        targets = [ex["tgt"] for ex in examples['translation']]

        inputs = [sentence_format(normalizer.normalize(text))
                for text in inputs]
        targets = [sentence_format(normalizer.normalize(text))
                for text in targets]
        
        model_inputs = tokenizer(
            inputs, max_length=max_input_length, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, truncation=True)

        if teacher_forcing: #right? 
            model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    @staticmethod
    def postprocess(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels


class M21RawTextProcessor(Processor):
            
    
    def preprocess(self,pairs_list, N, sacrebleu=True,
    #language_token_dict= config["language_token_dict"],
    tgt_tokens_in_src = False,
    token_conversion_dict = config["token_conversion_dict"],
    eval_languages = config['eval_languages']):
        """
        N: Samples per language, assuming languages are in the order found in config["eval_languages"]
        """

        sources = {}
        references = {}
        sacrereferences = {}

        datasets_list =  self.load_datasets( pairs_list, 
            token_conversion_dict, 
            validation_cutoff = N,mode = "cutoff_maximum")

        datasets_to_display = datasets.concatenate_datasets(datasets_list)

        for language, idx_begin in zip(eval_languages, range(0,len(datasets_to_display),N) ):
            if tgt_tokens_in_src:
                language_token = token_conversion_dict[language]
            sources[language] = [ text_pair["src"] for text_pair in datasets_to_display["translation"][idx_begin:idx_begin+N] ]
            if tgt_tokens_in_src:
                sources[language] = [language_token + ' ' +
                                s for s in sources[language]
                                ]
            else:
                sources[language] = [#language_token + ' ' +
                                s for s in sources[language]
                                ]
            references[language] = [ text_pair["tgt"] for text_pair in datasets_to_display["translation"][idx_begin:idx_begin+N] ]
            
            
            if sacrebleu:
                sacrereferences[language] = [ [s]
                                    for s in references[language]]
            
        if sacrebleu:
            return sources, references, sacrereferences
            
        return sources, references



class Many2OneProcessor(Processor):

    """
    Many2OneProcessor adds the src language token to the beginning of the src sentences, it has no target token
    """

    def load_datasets(self, pair_dicts_list, language_token_dict, validation_cutoff = 0,mode = "cutoff_maximum"):
        list_of_pairs = []
        for language_pair_dict in pair_dicts_list:
            src_language = language_pair_dict["source"]["language"]
            src_scentences = self.load_files(language_pair_dict["source"]["path"])
            tgt_language = language_pair_dict["target"]["language"]
            tgt_scentences = self.load_files(language_pair_dict["target"]["path"])
                    
            if validation_cutoff:
                if mode == "cutoff_maximum":
                    src_scentences = src_scentences[:validation_cutoff]
                    tgt_scentences = tgt_scentences[:validation_cutoff]
                elif mode == "cutoff_minimum":
                    src_scentences = src_scentences[validation_cutoff:]
                    tgt_scentences = tgt_scentences[validation_cutoff:]

            
            normalizer = sacremoses.MosesPunctNormalizer()

            src_scentences = [sentence_format(normalizer.normalize(text))
                    for text in src_scentences]
            tgt_scentences = [sentence_format(normalizer.normalize(text))
                    for text in tgt_scentences]
            
            src_scentences = [language_token_dict[src_language] + " " + src for src in src_scentences]
            #tgt_scentences = [language_token_dict[tgt_language] + " " + tgt for tgt in tgt_scentences]
            

            pairs = {'translation': [{"src": s,
                                "tgt": t}
                                for s, t in zip(src_scentences, tgt_scentences)]}

            list_of_pairs.append(datasets.Dataset.from_dict(pairs))

        return list_of_pairs

        @staticmethod
        def preprocess(examples, tokenizer):
            tokenizer.src_lang = ""
            tokenizer.tgt_lang = ""
            return super(Many2OneProcessor, Many2OneProcessor).preprocess(examples, tokenizer)


class Many2ManyProcessor(Processor):


    """
    Many2ManyProcessor adds the src language token to the beginning of the src sentences, and the tgt token 
    """
    @classmethod
    def _generate_pair_dataset(cls, src_path,src_language, tgt_path, tgt_language, validation_cutoff =0, mode="cutoff_minimum", marian_style_tokens = True):

        src_scentences = cls.load_files(src_path)
        tgt_scentences = cls.load_files(tgt_path)


        if validation_cutoff:
            if mode == "cutoff_maximum":
                src_scentences = src_scentences[:validation_cutoff]
                tgt_scentences = tgt_scentences[:validation_cutoff]
            elif mode == "cutoff_minimum":
                src_scentences = src_scentences[validation_cutoff:]
                tgt_scentences = tgt_scentences[validation_cutoff:]

        #TODO add tokens from tokenizer
        if marian_style_tokens:
            src_scentences = [tgt_language + " " + src for src in src_scentences]
            #tgt_scentences = [language_token_dict[tgt_language] + " " + tgt for tgt in tgt_scentences]


        pairs = {'translation': [{"src": s,
                            "tgt": t}
                            for s, t in zip(src_scentences, tgt_scentences)],
        }
        paired_dataset = datasets.Dataset.from_dict(pairs)
        paired_dataset.src_language = src_language
        paired_dataset.tgt_language = tgt_language
        
        return paired_dataset
    
    @classmethod
    def dataset_from_folders_m2m(cls,pair_dict, validation_cutoff = 0,mode = "cutoff_maximum"):

        list_of_paired_datasets = []
        for src_language in tqdm(pair_dict.keys()):
            for tgt_language in pair_dict[src_language].keys():
                if tgt_language == "all":
                    ### Get all other pairs for the same indicies
                    
                    for other_tgt_language in pair_dict.keys():
                        if other_tgt_language == src_language:
                            continue

                        for idx in range(len(pair_dict[src_language][tgt_language])):
                            src_path = pair_dict[src_language]["all"][idx]
                            tgt_path = pair_dict[other_tgt_language]["all"][idx]
                        
                            paired_dataset = cls._generate_pair_dataset(src_path,src_language, tgt_path, other_tgt_language, validation_cutoff = validation_cutoff, mode=mode)
                            list_of_paired_datasets.append(paired_dataset)
                    continue
                for idx in range(len(pair_dict[src_language][tgt_language])):
                    src_path = pair_dict[src_language][tgt_language][idx]
                    tgt_path = pair_dict[tgt_language][src_language][idx]
                    paired_dataset = cls._generate_pair_dataset(src_path,src_language, tgt_path, tgt_language, validation_cutoff = validation_cutoff, mode=mode)
                    list_of_paired_datasets.append(paired_dataset)

        return list_of_paired_datasets


    @staticmethod
    def iterative_preprocess(paired_datasets, tokenizer, mbart_style_tokens = True):
        tokenized_datasets = []
        for dataset_to_tokenize in tqdm(paired_datasets):
            if mbart_style_tokens:
                tokenizer.src_lang = "" # or src_language?            
                tokenizer.tgt_lang = ""
           
            tokenized_datasets.append(super(Many2ManyProcessor, Many2ManyProcessor).preprocess(dataset_to_tokenize, tokenizer))
        return tokenized_datasets


"""
tokenizer.src_lang = "ar_AR"
encoded_ar = tokenizer(article_ar, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)

#def tokenizer_call
#tokenizer_to)many2many(tokenizer):

    

class Many2ManyProcessor():

    def
"""
