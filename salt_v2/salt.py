'''Utilities for loading SALT v2 format data.'''
import json
import datasets

def translation_dataset(
    path,
    source_language,
    target_language,
    allow_target_language_in_source = True,
    prefix_target_language_in_source = False,
    languages_to_include = None,
    dataset_prefixes = []  ):
    '''Creates a translation dataset from a SALT v2 format source file.
 
    Various translation tasks, such as many-to-one and many-to-many, are catered
    for by setting `source_language` and `target_language`. For example:
        
    `source_language = 'en', target_language = 'lug'`     (English to Luganda)  
    `source_language = 'many', target_language = 'en'`    (many to English)
    `source_language = 'ach', target_language = 'many'`   (Acholi to many)
    `source_language = 'many', target_language = 'many'`  (many to many)
    `source_language = 'lug', target_language = 'lug'`    (pass-through from
                                                           monolingual text)
    
    Args:
        path : String containing the path to a .jsonl file.
        source_language: Either an ISO 639-2 language code (e.g. 'eng', 'lug'),
            or 'many' for multiple source languages.
        target_language: Either an ISO 639-2 language code (e.g. 'eng', 'lug'),
            or 'many' for multiple target languages.
        allow_target_language_in_source: In the case of many-to-one or
            many-to-many translation, specifies whether any records may contain
            the same language code for both source and target. For example, in
            many-to-English translation, can there be any records with source
            text also in English (default True).
        prefix_target_language_in_source: In the case of one-to-many and 
            many-to-many models where a prefixed language token can guide
            the model to know which language to generate as an output. (default: False)
        languages_to_include: If non-empty, a list of language codes
            that can be included in 'many'. Any text not in these languages will
            be ignored (Optional).
        dataset_prefixes: If non-empty, a list of dataset specific tokens to prepend to 
            sentences in the dataset. Ex. Backtranslation token, Out of domain token. 
            If used make sure to add the desired tokens to the tokenizer (Optional)
            
    Returns:
        dataset: A datasets.Dataset object with attributes `source`, `target`,
            `source_language` and `target_language`. The `source_language` and
            `target_language` attributes are always ISO 639-2 language codes
            and may vary from record to record, e.g. in the case of
            many-to-many.
    '''
    with open(path) as file:
        items = file.readlines()
        items = [json.loads(j)['text'] for j in items]

    dataset = {
        'source' : [],
        'target': [],
        'source_language': [],
        'target_language': [],
    }

    for item in items:
        all_language_codes = set(item.keys())

        if source_language == 'many':
            source_languages = all_language_codes
            if languages_to_include:
                source_languages = (
                    source_languages.intersection(languages_to_include))            
        else:
            source_languages = [source_language]

        if target_language == 'many':
            target_languages = all_language_codes
            if languages_to_include:
                target_languages = (
                    target_languages.intersection(languages_to_include))            
        else:
            target_languages = [target_language]


        for row_source_language in source_languages:
            for row_target_language in target_languages:
                
                if (row_source_language == row_target_language
                    and not allow_target_language_in_source):
                    continue
                if row_target_language not in item:
                    continue
                if row_source_language not in item:
                    continue
                
                source_sentence = item[row_source_language]

                if len(dataset_prefixes) > 0:
                    for prefix in dataset_prefixes: 
                        source_sentence = prefix + " " + source_sentence
                
                if prefix_target_language_in_source:
                    source_sentence = f">>{row_target_language}<<" + " " + source_sentence
                
                
                dataset['source'].append(source_sentence)
                
                dataset['target'].append(item[row_target_language])
                dataset['source_language'].append(row_source_language)
                dataset['target_language'].append(row_target_language)

    if not len(dataset):
        raise ValueError('No sentence pairs were found matching the specifications '
                         f'(path={path}, source_language={source_language}, '
                         f'target_language={target_language}, '
                         'allow_target_language_in_source='
                         f'{allow_target_language_in_source}, '
                         f'languages_to_include={languages_to_include}).')

    return datasets.Dataset.from_dict(dataset)