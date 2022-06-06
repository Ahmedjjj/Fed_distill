import unittest

import numpy as np 
from collections import Counter
import torch

from fed_distill.data.label_sampler import BalancedSampler, RandomSampler, WeightedSampler


class SamplerTestMethods(unittest.TestCase):
    def test_balanced_sampler(self):
        # Range classes
        classes = 10
        sampler = BalancedSampler(250, classes)
        target = Counter(list(range(classes)) * 25)
        self.assertEqual(target, Counter(next(iter(sampler)).tolist()))

        # Random classes
        classes = np.random.choice(range(50), size=10, replace=False)
        sampler = BalancedSampler(250, classes)
        target = Counter(classes.tolist() * 25)
        self.assertEqual(target, Counter(next(iter(sampler)).tolist()))

    def test_random_sampler(self):
        classes = 10
        sampler = RandomSampler(250, classes)
        self.assertEqual(set(range(classes)), set(next(iter(sampler)).tolist()))

        # Random classes
        classes = np.random.choice(range(50), 10, replace=True)
        sampler = RandomSampler(250, classes)
        self.assertEqual(set(classes), set(next(iter(sampler)).tolist()))
    
        # Probabililities
        classes = np.random.choice(range(50), 10, replace=True)
        sampler_iter = iter(RandomSampler(int(1e6), classes))
        result = torch.cat(tuple(next(sampler_iter) for _ in range(20))).tolist()
        label_counts = Counter(result).values()
        
        for count in label_counts:
            self.assertAlmostEqual(1 / 10, count / len(result), places=1)        
    
        
if __name__ == "__main__":
    unittest.main()