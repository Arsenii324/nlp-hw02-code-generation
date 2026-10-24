"""Behavior check for the reusable measured beam inference wrapper."""
import importlib.util
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from nlp_hw02.model import load_model


def main():
    assert importlib.util.find_spec('nlp_hw02.beam_inference') is not None,'Missing reusable beam inference wrapper'
    from nlp_hw02.beam_inference import generate_beam_questions
    spec=importlib.util.spec_from_file_location('measured_gpu_profile','scripts/profile_decoding_gpu.py')
    measured=importlib.util.module_from_spec(spec);spec.loader.exec_module(measured)
    torch.set_num_threads(2)
    root=Path('artifacts/kaggle/b005-b006-v1/run-b005')
    model,_=load_model(root/'model.pt',device='cpu');tok=Tokenizer.from_file(str(root/'tokenizer.json'))
    questions=[r['question'] for r in json.loads((root/'validation.json').read_text())[:3]]
    expected,_=measured.beam_questions(model,tok,questions,batch_size=2)
    actual,timing=generate_beam_questions(model,tok,questions,batch_size=2)
    assert actual==expected and timing['rows']==3 and timing['beam_width']==4 and timing['length_penalty']==.6
    assert timing['seconds']>0 and timing['capped_rows']==0
    for batch in (0,-1):
        try: generate_beam_questions(model,tok,questions,batch_size=batch)
        except ValueError: pass
        else: raise AssertionError('Nonpositive batch accepted')
    empty,timing=generate_beam_questions(model,tok,[],batch_size=2)
    assert empty==[] and timing['rows']==0 and timing['mean_generated_tokens']==0
    print('PASS reusable beam outputs match measured wrapper; partial batch; empty input; invalid batch')


if __name__=='__main__': main()
