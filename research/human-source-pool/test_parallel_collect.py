import unittest
from parallel_collect import validate_jobs,alive

class ParallelTests(unittest.TestCase):
    def test_duplicate_source_jobs_rejected(self):
        j={'source_id':'imdb','script':'ingest.py'}
        with self.assertRaises(ValueError):validate_jobs([j,j],{'imdb':1061})
    def test_removed_source_rejected(self):
        with self.assertRaises(ValueError):validate_jobs([{'source_id':'yelp','script':'ingest.py'}],{'imdb':1061})
    def test_script_path_escape_rejected(self):
        with self.assertRaises(ValueError):validate_jobs([{'source_id':'imdb','script':'../wrong.py'}],{'imdb':1061})
    def test_stale_process_not_alive(self):
        self.assertFalse(alive({'pid':999999999,'start_ticks':'0'}))

if __name__=='__main__':unittest.main()
