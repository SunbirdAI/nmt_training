import datasets 

from tqdm import tqdm 

def _file_to_list(path):
    with open(path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        return lines

def dataset_from_folders(pair_dicts_list, language_token_dict, validation_cutoff = 0,mode = "cutoff_maximum"):

    list_of_pairs = []
    for language_pair_dict in pair_dicts_list:
        src_language = language_pair_dict["source"]["language"]
        src_scentences = _file_to_list(language_pair_dict["source"]["path"])
        tgt_language = language_pair_dict["target"]["language"]
        tgt_scentences = _file_to_list(language_pair_dict["target"]["path"])
                
        if validation_cutoff:
            if mode == "cutoff_maximum":
                src_scentences = src_scentences[:validation_cutoff]
                tgt_scentences = tgt_scentences[:validation_cutoff]
            elif mode == "cutoff_minimum":
                src_scentences = src_scentences[validation_cutoff:]
                tgt_scentences = tgt_scentences[validation_cutoff:]

        # pairs = {'translation': [{src_language: s,
        #                     tgt_language: t}
        #                      for s, t in zip(src_scentences, tgt_scentences)]}
        
        src_scentences = [language_token_dict[src_language] + " " + src for src in src_scentences]
        tgt_scentences = [language_token_dict[tgt_language] + " " + tgt for tgt in tgt_scentences]
        

        pairs = {'translation': [{"src": s,
                            "tgt": t}
                             for s, t in zip(src_scentences, tgt_scentences)]}

        list_of_pairs.append(datasets.Dataset.from_dict(pairs))

    return list_of_pairs

# config['training_subset_paths'] = [
#             "ach":{
#                 "all":[config['data_dir'] + "v7.0/supervised/en-ach/train.ach"],
#                 "en":[],
#                 "lgg":[],
#                 "lug":[],
#                 "nyn":[],
#                 "teo":[]


def _generate_pair_dataset(src_path,src_language, tgt_path, tgt_language, validation_cutoff =0, mode="cutoff_minimum"):
    src_scentences = _file_to_list(src_path)
    tgt_scentences = _file_to_list(tgt_path)


    if validation_cutoff:
        if mode == "cutoff_maximum":
            src_scentences = src_scentences[:validation_cutoff]
            tgt_scentences = tgt_scentences[:validation_cutoff]
        elif mode == "cutoff_minimum":
            src_scentences = src_scentences[validation_cutoff:]
            tgt_scentences = tgt_scentences[validation_cutoff:]

    #TODO add tokens from tokenizer
    #src_scentences = [language_token_dict[src_language] + " " + src for src in src_scentences]
    #tgt_scentences = [language_token_dict[tgt_language] + " " + tgt for tgt in tgt_scentences]


    pairs = {'translation': [{"src": s,
                        "tgt": t}
                        for s, t in zip(src_scentences, tgt_scentences)],
    }
                        #"src_language":src_language,
                        #"tgt_language":tgt_language}
    paired_dataset = datasets.Dataset.from_dict(pairs)
    paired_dataset.src_language = src_language
    paired_dataset.tgt_language = tgt_language
    
    return paired_dataset

def dataset_from_folders_m2m(pair_dict, validation_cutoff = 0,mode = "cutoff_maximum"):

    list_of_paired_datasets = []
    for src_language in tqdm(pair_dict.keys()):
        for tgt_language in pair_dict[src_language].keys():
            if tgt_language == "all":
                ### Get all other pairs for the same indicies
                
                for other_tgt_language in pair_dict.keys():
                    if other_tgt_language == src_language:
                        continue

                    for idx in range(len(pair_dict[src_language][tgt_language])):
                        print("all")
                        src_path = pair_dict[src_language]["all"][idx]
                        tgt_path = pair_dict[other_tgt_language]["all"][idx]
                    
                        paired_dataset = _generate_pair_dataset(src_path,src_language, tgt_path, other_tgt_language, validation_cutoff = validation_cutoff, mode=mode)
                        list_of_paired_datasets.append(paired_dataset)
                continue
            for idx in range(len(pair_dict[src_language][tgt_language])):
                print("specific")
                src_path = pair_dict[src_language][tgt_language][idx]
                tgt_path = pair_dict[tgt_language][src_language][idx]
                paired_dataset = _generate_pair_dataset(src_path,src_language, tgt_path, tgt_language, validation_cutoff = validation_cutoff, mode=mode)
                list_of_paired_datasets.append(paired_dataset)

    return list_of_paired_datasets
