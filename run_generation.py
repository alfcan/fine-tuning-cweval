#!/usr/bin/env python3
import sys
import os
import multiprocessing

# Set spawn as the start method globally to avoid macOS fork locks when using p_map
try:
    multiprocessing.set_start_method('spawn', force=True)
except Exception:
    pass

# Add CWEval directory to python path
sys.path.insert(0, os.path.abspath("CWEval"))
from cweval.generate import Gener
import fire

# We subclass Gener to bypass the multiprocessing pool when num_proc <= 1, 
# preventing any deadlock or fork safety locks.
class SafeGener(Gener):
    def gen(self) -> None:
        if self.num_proc <= 1:
            print("Running sequential generation via SafeGener wrapper...")
            from tqdm import tqdm
            for case in tqdm(self.cases.values()):
                self._gen_case(self.model, self.ppt, case, self.ai_kwargs, 0)
        else:
            super().gen()

if __name__ == "__main__":
    fire.Fire(SafeGener)
