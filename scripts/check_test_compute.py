"""Focused contract checks for reference-free numeric plurality selection."""
from test_compute_probe import select_plurality

codes=['def solve():\n return 2','def solve():\n return 1+1','def solve():\n return 3']
index,info=select_plurality(codes,[-3.,-1.,-.1])
assert index==1 and info['winning_votes']==2 and info['supported_candidates']==3
index,info=select_plurality(['def solve():\n return x','def solve():\n return 4'],[-.1,-2.])
assert index==1 and info['supported_candidates']==1
index,info=select_plurality(['broken','def solve():\n return x'],[-1.,-2.])
assert index==0 and info['supported_candidates']==0
index,_=select_plurality(['def solve():\n return 2','def solve():\n return 3'],[-2.,-1.])
assert index==1
try:select_plurality([],[])
except ValueError:pass
else:raise AssertionError('Empty candidates accepted')
print('REFERENCE_FREE_PLURALITY_CHECKS_OK')
