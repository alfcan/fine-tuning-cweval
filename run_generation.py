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
    def __init__(
        self,
        eval_path: str = '',
        model: str = 'gpt-4o-mini-2024-07-18',
        ppt: str = 'direct',
        num_proc: int = 8,
        langs = None,
        exclude_path = None,
        include_path = None,
        # AI parameters
        n: int = 20,
        max_completion_tokens: int = 2048,
        temperature: float = 0.8,
        **kwargs,
    ):
        import os
        orig_exists = os.path.exists
        
        def mock_exists(path):
            if eval_path and os.path.abspath(path) == os.path.abspath(eval_path):
                return False
            return orig_exists(path)
            
        os.path.exists = mock_exists
        try:
            if langs is None:
                from cweval.commons import LANGS
                langs = LANGS
            if exclude_path is None:
                exclude_path = []
            if include_path is None:
                include_path = []
                
            super().__init__(
                eval_path=eval_path,
                model=model,
                ppt=ppt,
                num_proc=num_proc,
                langs=langs,
                exclude_path=exclude_path,
                include_path=include_path,
                n=n,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                **kwargs
            )
        finally:
            os.path.exists = orig_exists

    @staticmethod
    def _gen_case(
        ai: str,
        ppt: str,
        case: dict,
        ai_kwargs: dict,
        rank: int,
    ) -> None:
        import os
        from cweval.ai import AIAPI
        from cweval.ppt import make_prompt

        num_samples = ai_kwargs.get('n', 1)
        missing_indices = []
        for i in range(num_samples):
            out_path = case['out_path_template'].format(index=i)
            if not os.path.exists(out_path):
                missing_indices.append(i)
        
        if not missing_indices:
            print(f"{case['task_file_path']} already has all {num_samples} samples, skipping", flush=True)
            return

        print(f"Generating {len(missing_indices)} missing samples for {case['task_file_path']}...")
        
        local_kwargs = ai_kwargs.copy()
        local_kwargs['n'] = len(missing_indices)
        
        aiapi = AIAPI(ai, **local_kwargs)
        prompt = make_prompt(ppt)
        resps = prompt.req_ai(
            aiapi,
            case['lang'],
            case['code_prompt'],
            metadata={
                k: v for k, v in case.items() if k not in ['code_prompt', 'lang']
            },
        )
        
        for idx, resp in zip(missing_indices, resps):
            out_path = case['out_path_template'].format(index=idx)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                f.write(resp)

    def gen(self) -> None:
        if self.num_proc <= 1:
            print("Running sequential generation via SafeGener wrapper...")
            from tqdm import tqdm
            for case in tqdm(self.cases.values()):
                self._gen_case(self.model, self.ppt, case, self.ai_kwargs, 0)
        else:
            print(f"Running parallel generation via ThreadPoolExecutor with {self.num_proc} threads...")
            from concurrent.futures import ThreadPoolExecutor
            from tqdm import tqdm
            
            cases_list = list(self.cases.values())
            with ThreadPoolExecutor(max_workers=self.num_proc) as executor:
                futures = [
                    executor.submit(
                        self._gen_case,
                        self.model,
                        self.ppt,
                        case,
                        self.ai_kwargs,
                        idx
                    )
                    for idx, case in enumerate(cases_list)
                ]
                for future in tqdm(futures, desc="Generating samples"):
                    future.result()

if __name__ == "__main__":
    fire.Fire(SafeGener)
