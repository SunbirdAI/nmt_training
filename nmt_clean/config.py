import torch
import transformers

# Parameters for mul-en models
config = {
    'source_languages': ["ach", "lgg", "lug", "nyn", "teo"],
    'target_languages': ['en'],
    'metric_for_best_model': 'loss',
    'train_batch_size': 1,
    'gradient_accumulation_steps': 2400,
    'max_input_length': 128,
    'max_target_length': 128,
    'validation_samples_per_language': 100,
    'testing_samples_per_language': 100,
    'validation_train_merge': True,
    'eval_batch_size': 1,
    'eval_languages': ["ach", "lgg", "lug", "nyn", "teo"],
    'eval_pretrained_model': False,
    'learning_rate': 1e-4,
    'num_train_epochs': 2,
    'label_smoothing_factor': 0.1,
    'flores101_training_data': True,
    'mt560_training_data': True,
    'back_translation_training_data': False,
    'front_translation_training_data': False, #not implemented
    'named_entities_training_data': False,
    'recycle_language_tokens': True,
    'google_back_translation': True,
    'oversample_rate': 5,
    'oversample_in_domain': True
}

config['language_pair'] = f'salt-en'
config['wandb_project'] = f'salt-mbart'
config['wandb_entity'] = f'sunbird'

config['model_checkpoint'] = f'/content/mBART_weights/best_mBART_salt'

# What training data to use
config['data_dir'] = f'/home/ali/Documents/repos/datasets/salt/v7-dataset/'
config['training_data_dir'] = f'v7-dataset/v7.0/supervised/mul-en/'
config['validation_data_dir'] = f'v7-dataset/v7.0/supervised/mul-en/'
config['test_data_dir'] = f'v7-dataset/v7.0/supervised/mul-en/'

config['training_extra_data_dir'] = f'v7-dataset/v7.0/supervised/mul-en/'

# Evaluate roughly every 10 minutes
# eval_steps_interval = 350 * 60 * 7 / (config['gradient_accumulation_steps']
#                                       * config['train_batch_size'])

eval_steps_interval = 20#4 * max(1, int(eval_steps_interval / 10))

print(f'Evaluating every {eval_steps_interval} training steps.')

config['train_settings'] = transformers.Seq2SeqTrainingArguments(
    f'output-{config["language_pair"]}',
    evaluation_strategy = 'steps',
    eval_steps = eval_steps_interval,
    save_steps = eval_steps_interval,
    gradient_accumulation_steps = config['gradient_accumulation_steps'],
    learning_rate = config['learning_rate'],
    per_device_train_batch_size = config['train_batch_size'],
    per_device_eval_batch_size = config['eval_batch_size'],
    weight_decay = 0.01,
    save_total_limit = 3,
    num_train_epochs = config['num_train_epochs'],
    predict_with_generate = True,
    fp16 = torch.cuda.is_available(),
    logging_dir = f'output-{config["language_pair"]}',
    report_to = 'none',
    run_name = f'{config["language_pair"]}',
    load_best_model_at_end=True,
    metric_for_best_model = config['metric_for_best_model'],
    label_smoothing_factor = config['label_smoothing_factor'],
    
)

## Be careful to keep the order the same for source and target dataset pairs
config['training_subset_paths'] = [
        {
            "source":{"language":"ach",
                   "path":config['data_dir'] + "v7.0/supervised/en-ach/train.ach"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/train.en"} 
        },
        {
            "source":{"language":"lgg",
                   "path":config['data_dir'] + "v7.0/supervised/en-lgg/train.lgg"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/train.en"} 
        },
        {
            "source":{"language":"lug",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/train.lug"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/train.en"} 
        },
        {
            "source":{"language":"nyn",
                   "path":config['data_dir'] + "v7.0/supervised/en-nyn/train.nyn"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/train.en"} 
        },
        
        {
            "source":{"language":"teo",
                   "path":config['data_dir'] + "v7.0/supervised/en-teo/train.teo"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/train.en"} 
        }
    
   ]


    
config['validation_subset_paths'] = [
        
        {
            "source":{"language":"ach",
                   "path":config['data_dir'] + "v7.0/supervised/en-ach/val.ach"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/val.en"} 
        },
        {
            "source":{"language":"lgg",
                   "path":config['data_dir'] + "v7.0/supervised/en-lgg/val.lgg"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/val.en"} 
        },
        {
            "source":{"language":"lug",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/val.lug"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/val.en"} 
        },
        {
            "source":{"language":"nyn",
                   "path":config['data_dir'] + "v7.0/supervised/en-nyn/val.nyn"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/val.en"} 
        },
        {
            "source":{"language":"teo",
                   "path":config['data_dir'] + "v7.0/supervised/en-teo/val.teo"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/en-lug/val.en"} 
        }
    
   ]


config['testing_subset_paths'] = [
        
        {
            "source":{"language":"ach",
                   "path":config['data_dir'] + "v7.0/supervised/mul-en/test_ach.src"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/mul-en/test_ach.tgt"} 
        },
        {
            "source":{"language":"lgg",
                   "path":config['data_dir'] + "v7.0/supervised/mul-en/test_lgg.src"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/mul-en/test_lgg.tgt"} 
        },
        {
            "source":{"language":"lug",
                   "path":config['data_dir'] + "v7.0/supervised/mul-en/test_lug.src"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/mul-en/test_lug.tgt"} 
        },
        {
            "source":{"language":"nyn",
                   "path":config['data_dir'] + "v7.0/supervised/mul-en/test_nyn.src"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/mul-en/test_nyn.tgt"} 
        },
        {
            "source":{"language":"teo",
                   "path":config['data_dir'] + "v7.0/supervised/mul-en/test_teo.src"},
            "target":{"language":"en",
                   "path":config['data_dir'] + "v7.0/supervised/mul-en/test_teo.tgt"} 
        }
    
   ]

#why not luo?
if config['flores101_training_data']:
    flores_dict = {
        "source":{
            "language":"lug",
            "path":config['data_dir'] + "v7.0/supervised/mul-en/train_flores_lug.src"
        },
        "target":{
            "language":"en",
            "path":config['data_dir'] + "v7.0/supervised/mul-en/train_flores_lug.tgt"
        }
    }
    config['training_subset_paths'].append(flores_dict)

# if config['back_translation_training_data']:
#     raise NotImplementedError("Have not split bt data by language yet")
#     config['training_subset_ids'].append('back_translated')

# Over-sample the non-religious training text
#config['training_subset_ids'] = config['training_subset_ids'] * 5
# Will oversample from interleave datasets

#if config['back_translation_training_data']:
#     raise NotImplementedError("Have not split bt data by language yet")
#     config['training_subset_ids'].append('back_translated')

if config["google_back_translation"]:
    google_bt = {
        "source":{
            "language":"lug",
            "path":config['data_dir'] + "v7.0/supervised/mul-en/bukedde_ggl_bt_lug.src"
        },
        "target":{
            "language":"en",
            "path":config['data_dir'] + "v7.0/supervised/mul-en/bukedde_ggl_bt_lug.src"
        }
    }
    config['training_subset_ids'].append(google_bt)

if config["oversample_in_domain"]:
    config['training_subset_ids'] = config['training_subset_ids'] * config["oversample_rate"]


if config['mt560_training_data']:
    mt560_list = [
        {
            "source":{
            "language":"ach",
            "path":config['data_dir'] + "v7.0/supervised/mul-en/train_mt560_ach.src"
            },
            "target":{
                "language":"en",
                "path":config['data_dir'] + "v7.0/supervised/mul-en/train_mt560_ach.tgt"
            }  
        },
        {
            "source":{
            "language":"lug",
            "path":config['data_dir'] + "v7.0/supervised/mul-en/train_mt560_lug.src"
            },
            "target":{
                "language":"en",
                "path":config['data_dir'] + "v7.0/supervised/mul-en/train_mt560_lug.tgt"
            }
        },
        {
            "source":{
            "language":"nyn",
            "path":config['data_dir'] + "v7.0/supervised/mul-en/train_mt560_nyn.src"
            },
            "target":{
                "language":"en",
                "path":config['data_dir'] + "v7.0/supervised/mul-en/train_mt560_nyn.tgt"
            }
        }        

    ]
    config['training_subset_paths'].extend(mt560_list)

# if config['named_entities_training_data']:
#     rasie NotImplementedError("NER pairs are aggregate not separate")
#     config['training_subset_ids'].append('named_entities')

if config["recycle_language_tokens"]:
    config["token_conversion_dict"] = {
        "teo": 'ar_AR' ,
        "ach": 'cs_CZ',
        "lug": 'de_DE',
        "lgg": 'es_XX',
        "nyn": 'et_EE',
        "en": 'en_XX'
        
     }
else:
    raise NotImplementedError("Code to add tokens and resize embedding layer not added")
    # If you want to add it refer to https://www.depends-on-the-definition.com/how-to-add-new-tokens-to-huggingface-transformers/