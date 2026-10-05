# chunk-bench results

Ran 24 gold questions over 12 documents, k=5.

Best strategy on recall@5: **fixed-1000** (recall=1.000, MRR=0.917).

Full table:

```
strategy        recall@k  mrr    context_precision@k  n_chunks  avg_chunk_chars
-------------------------------------------------------------------------------
fixed-1000      1.000     0.917  0.325                24        795            
fixed-2000      1.000     0.958  0.200                12        1489           
sentence-1000   1.000     0.958  0.325                24        803            
sentence-2000   1.000     0.958  0.200                12        1482           
fixed-500       0.958     0.799  0.433                46        425            
heading-1000    0.958     0.922  0.425                49        362            
heading-2000    0.958     0.922  0.425                49        362            
```
