"""Resident and mmap arrays produce exactly the same seeded training samples."""
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
p=Path(__file__).resolve().parents[1]/'src/nlp_hw02/training_arrays.py';assert p.exists(),'Missing bounded training array loader'
s=importlib.util.spec_from_file_location('arrays',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
with tempfile.TemporaryDirectory() as d:
    root=Path(d);rng=np.random.default_rng(42)
    a=rng.integers(0,2048,(123,7),dtype=np.uint16);b=rng.integers(0,2048,(123,5),dtype=np.uint16)
    np.save(root/'train_source.npy',a);np.save(root/'train_target.npy',b)
    events=[];aa,bb,report=m.load_training_arrays(root,resident=True,available_bytes=10**10,phase=lambda name:events.append(name))
    ma,mb,_=m.load_training_arrays(root,resident=False)
    assert not isinstance(aa,np.memmap) and isinstance(ma,np.memmap)
    assert report['resident_bytes']==a.nbytes+b.nbytes and events==['load_source','load_target']
    first=np.random.default_rng(42);second=np.random.default_rng(42)
    for _ in range(4):
        i=first.integers(0,len(a),64);j=second.integers(0,len(a),64)
        assert np.array_equal(i,j) and np.array_equal(aa[i],ma[j]) and np.array_equal(bb[i],mb[j])
    try:m.load_training_arrays(root,resident=True,available_bytes=1)
    except RuntimeError:pass
    else:raise AssertionError('Insufficient RAM admitted')
print('PASS identical arrays/seeded batches, resident versus mmap types, phase boundaries, bounded RAM')
