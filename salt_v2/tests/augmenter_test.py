import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import json
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import pandas as pd
import salt 
import shutil
import tempfile
import unittest

class TestAugmentationMethods(unittest.TestCase):

    @staticmethod
    def _write_to_jsonl_file(data, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            for d in data:
                json.dump(d, f, ensure_ascii=False)
                f.write('\n')

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_jsonl_path = self.test_dir + '/test.jsonl'

        data = [
            {"text": {"en": "Hello, how are you?", "es": "Hola, ¿cómo estás?"}},
            {"text": {"en": "I love pizza", "it": "Adoro la pizza"}},
            {"text": {"fr": "Bonjour tout le monde", "es": "Hola a todos"}},
            {"text": {"de": "Ich spreche kein Deutsch", "en": "I don't speak German"}}
        ]

        self._write_to_jsonl_file(data, self.test_jsonl_path)
        aug = naw.RandomWordAug(aug_min=1, aug_max=1)
        self.delete_word = aug.augment
        self.random_augmentations = naf.Sequential([nac.RandomCharAug(), naw.RandomWordAug()])

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_translation_count_1(self):
        # Test case 1: keep_unaugmented_src = False, target_augmenter = None
        dataset = salt.translation_dataset(
            path=self.test_jsonl_path, 
            source_language='en', 
            target_language='it',
            allow_target_language_in_source=True,
            prefix_target_language_in_source=False,
            languages_to_include=None,
            dataset_prefixes=[],
            source_augmenter=None,
            keep_unaugmented_src=False,
            target_augmenter=None
        )
        assert len(dataset['source']) == len(dataset['target'])
    
    def test_translation_count_2(self):
        # Test case 2: keep_unaugmented_src = True, target_augmenter = None
        dataset = salt.translation_dataset(
            path=self.test_jsonl_path, 
            source_language='en', 
            target_language='it',
            allow_target_language_in_source=True,
            prefix_target_language_in_source=False,
            languages_to_include=None,
            dataset_prefixes=[],
            source_augmenter=None,
            keep_unaugmented_src=True,
            target_augmenter=None
        )
        assert len(dataset['source']) == len(dataset['target']) 
    
    def test_translation_count_3(self):
        # Test case 3: keep_unaugmented_src = False, target_augmenter is a function that repeats the sentence twice
        dataset = salt.translation_dataset(
            path=self.test_jsonl_path, 
            source_language="en", 
            target_language="en",
            allow_target_language_in_source=True,
            prefix_target_language_in_source=False,
            languages_to_include=None,
            dataset_prefixes=[],
            source_augmenter=self.random_augmentations.augment,
            keep_unaugmented_src=False,
            target_augmenter=None
        )
        #print(dataset['source'])
        #print(dataset['target'])

        assert len(dataset['source']) == len(dataset['target'])
        
    def test_translation_count_4(self):
        # Test case 4: keep_unaugmented_src = True, target_augmenter is a function that replaces every other word with 'test'
        dataset = salt.translation_dataset(
            path=self.test_jsonl_path, 
            source_language='en', 
            target_language='it',
            allow_target_language_in_source=True,
            prefix_target_language_in_source=False,
            languages_to_include=None,
            dataset_prefixes=[],
            source_augmenter=self.random_augmentations.augment,
            keep_unaugmented_src=True,
            target_augmenter=None
        )
        
        assert len(dataset['source']) == len(dataset['target']) 

    def test_example_augmentation(self):
        text = "The quick brown fox jumps over the lazy dog."
        expected_augmented_text = [
            "Quick brown fox jumps over the lazy dog.",
            "The brown fox jumps over the lazy dog.",
            "The quick fox jumps over the lazy dog.",
            "The quick brown jumps over the lazy dog.",
            "The quick brown fox over the lazy dog.",
            "The quick brown fox jumps the lazy dog.",
            "The quick brown fox jumps over lazy dog.",
            "The quick brown fox jumps over the dog.",
            "The quick brown fox jumps over the lazy."
        ]
        augmented_text = self.delete_word(text)[0]
        
        #print(f"Random Word Augmentation: {augmented_text}")
        assert augmented_text in expected_augmented_text

if __name__ == '__main__':
    unittest.main()
