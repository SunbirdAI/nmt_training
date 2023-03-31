import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

import pandas as pd
import salt
import shutil
import tempfile
import unittest


class TestDataLoadingMethods(unittest.TestCase):
  def setUp(self):
    self.test_dir = tempfile.mkdtemp()
    self.test_jsonl_path = self.test_dir + '/test.jsonl'
    
    with open(self.test_jsonl_path, 'w') as f:
        f.write(
          '{"text" : {"lug" : "lug1", "ach" : "ach1", "eng" : "eng1"}}\n'
          '{"text" : {"lug" : "lug2", "ach" : "ach2", "eng" : "eng2"}}'
        )
    
  def tearDown(self):
      shutil.rmtree(self.test_dir)

  def test_one_to_one(self):
      dataset = salt.translation_dataset(
          path=self.test_jsonl_path,
          source_language = 'lug',
          target_language = 'ach')
      dataset = pd.DataFrame(dataset.to_dict()).to_dict(orient='records')
      expected = {
        'source': ['lug1', 'lug2'],
        'target': ['ach1', 'ach2'],
        'source_language': ['lug', 'lug'],
        'target_language': ['ach', 'ach']}
      expected = pd.DataFrame(expected).to_dict(orient='records')
      self.assertCountEqual(dataset, expected)
      
  def test_one_to_many(self):
      dataset = salt.translation_dataset(
          path=self.test_jsonl_path,
          source_language = 'lug',
          target_language = 'many')
      dataset = pd.DataFrame(dataset.to_dict()).to_dict(orient='records')
      expected = {
        'source': ['lug1', 'lug1', 'lug1', 'lug2', 'lug2', 'lug2'],
        'target': ['eng1', 'lug1', 'ach1', 'eng2', 'lug2', 'ach2'],
        'source_language': ['lug', 'lug', 'lug', 'lug', 'lug', 'lug'],
        'target_language': ['eng', 'lug', 'ach', 'eng', 'lug', 'ach']}
      expected = pd.DataFrame(expected).to_dict(orient='records')
      self.assertCountEqual(dataset, expected)

  def test_many_to_one(self):
      dataset = salt.translation_dataset(
          path=self.test_jsonl_path,
          source_language = 'many',
          target_language = 'eng')
      dataset = pd.DataFrame(dataset.to_dict()).to_dict(orient='records')
      expected = {
        'source': ['eng1', 'lug1', 'ach1', 'eng2', 'lug2', 'ach2'],
        'target': ['eng1', 'eng1', 'eng1', 'eng2', 'eng2', 'eng2'], 
        'source_language': ['eng', 'lug', 'ach', 'eng', 'lug', 'ach'], 
        'target_language': ['eng', 'eng', 'eng', 'eng', 'eng', 'eng']}
      expected = pd.DataFrame(expected).to_dict(orient='records')
      self.assertCountEqual(dataset, expected)

  def test_many_to_many(self):
      dataset = salt.translation_dataset(
          path=self.test_jsonl_path,
          source_language = 'many',
          target_language = 'many')
      dataset = pd.DataFrame(dataset.to_dict()).to_dict(orient='records')
      expected = {
        'source': ['lug1', 'lug1', 'lug1', 'ach1', 'ach1', 'ach1',
                   'eng1', 'eng1', 'eng1', 'lug2', 'lug2', 'lug2',
                   'ach2', 'ach2', 'ach2', 'eng2', 'eng2', 'eng2'], 
        'target': ['lug1', 'ach1', 'eng1', 'lug1', 'ach1', 'eng1',
                   'lug1', 'ach1', 'eng1', 'lug2', 'ach2', 'eng2',
                   'lug2', 'ach2', 'eng2', 'lug2', 'ach2', 'eng2'], 
        'source_language': ['lug', 'lug', 'lug', 'ach', 'ach', 'ach',
                            'eng', 'eng', 'eng', 'lug', 'lug', 'lug',
                            'ach', 'ach', 'ach', 'eng', 'eng', 'eng'], 
        'target_language': ['lug', 'ach', 'eng', 'lug', 'ach', 'eng',
                            'lug', 'ach', 'eng', 'lug', 'ach', 'eng',
                            'lug', 'ach', 'eng', 'lug', 'ach', 'eng']}
      expected = pd.DataFrame(expected).to_dict(orient='records')
      self.assertCountEqual(dataset, expected)
      
if __name__ == '__main__':
    unittest.main()