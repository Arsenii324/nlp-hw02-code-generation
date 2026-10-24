"""Load existing immutable uint16 token arrays without changing their samples."""
import os
from pathlib import Path
import time
import numpy as np


def available_memory_bytes():
    values=[]
    info=Path('/proc/meminfo')
    if info.exists():
        row=next(line for line in info.read_text().splitlines() if line.startswith('MemAvailable:'))
        values.append(int(row.split()[1])*1024)
    for limit,used in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                       ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            maximum=Path(limit).read_text().strip()
            if maximum!='max':values.append(max(0,int(maximum)-int(Path(used).read_text())))
        except (OSError,ValueError):pass
    if not values:
        try:values.append(os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'))
        except (ValueError,OSError):raise RuntimeError('Cannot establish available RAM for resident arrays')
    return min(values)


def array_resident_bytes(root):
    arrays=[np.load(Path(root)/n,mmap_mode='r') for n in ('train_source.npy','train_target.npy')]
    assert all(a.dtype==np.uint16 and a.ndim==2 and a.flags.c_contiguous for a in arrays)
    assert len(arrays[0])==len(arrays[1])
    return sum(a.nbytes for a in arrays)


def load_training_arrays(root,resident=False,available_bytes=None,phase=None):
    root=Path(root);size=array_resident_bytes(root)
    available=available_memory_bytes() if resident and available_bytes is None else available_bytes
    # Allow one process's arrays plus page cache and1GiB headroom. Parent admits
    # both processes together before launch; this is a second check at allocation.
    if resident and available<2*size+1024**3:
        raise RuntimeError(f'Insufficient RAM: available{available}, resident{size}')
    arrays=[];timings={}
    for name,filename in [('load_source','train_source.npy'),('load_target','train_target.npy')]:
        if phase is not None:phase(name)
        start=time.monotonic();arrays.append(np.load(root/filename,mmap_mode=None if resident else 'r'))
        timings[name+'_seconds']=time.monotonic()-start
    return *arrays,{'mode':'resident' if resident else 'mmap','resident_bytes':size if resident else 0,
                   'available_bytes_before':available,**timings}
