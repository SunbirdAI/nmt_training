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

