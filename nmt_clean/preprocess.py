import datasets
import sacremoses

from nmt_clean.config import config 

def sentence_format(input):
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
    def preprocess(examples, tokenizer):
        normalizer = sacremoses.MosesPunctNormalizer()
        inputs = [ex["src"] for ex in examples['translation']]
        targets = [ex["tgt"] for ex in examples['translation']]

        inputs = [sentence_format(normalizer.normalize(text))
                for text in inputs]
        targets = [sentence_format(normalizer.normalize(text))
                for text in targets]
        
        model_inputs = tokenizer(
            inputs, max_length=config['max_input_length'], truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=config['max_target_length'], truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    @staticmethod
    def postprocess(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels


class M21RawTextProcessor(Processor):
            
    
    def preprocess(pairs_list, N, sacrebleu=True):
        """
        N: Samples per language, assuming languages are in the order found in config["eval_languages"]
        """

        sources = {}
        references = {}
        sacrereferences = {}

        datasets_list =  self.load_datasets( pairs_list, 
            config["language_token_dict"], 
            validation_cutoff = N,mode = "cutoff_maximum")

        datasets_to_display = datasets.concatenate_datasets(datasets_list)

        for language, idx_begin in zip(config['eval_languages'], range(0,len(datasets_to_display),N) ):
            language_tokem = config["token_conversion_dict"][language]
            sources[language] = [ text_pair["src"] for text_pair in datasets_to_display["translation"][idx_begin:idx_begin+N] ]
            sources[language] = [language_tokem + ' ' + s
                                for s in sources[language]]
            references[language] = [ text_pair["tgt"] for text_pair in datasets_to_display["translation"][idx_begin:idx_begin+N] ]
            
            
        if sacrebleu:
            sacrereferences[language] = [ [s]
                                    for s in references[language]]
            
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