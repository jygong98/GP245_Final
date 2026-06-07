# Benchmark: test1 (1-layer half-space + Moho)

Upstream example: `ref_code/FDFK2D-master_experiment/examples/test1`

| Parameter | Value |
|-----------|-------|
| Profile | 0–200 km, depth 0–50 km |
| Grid | nx=1001, nz=251, dx=dz=200 m |
| FD order | 16 (half-order 8 in code) |
| Time | tmax=40 s, dt=0.01 s → nt=4001 |
| Receivers | surface array (see `input/Receiver.dat`) |

## Submit

```bash
cd benchmark/test1
sbatch submit.sbatch
```

Results: `output/benchmark/benchmark_summary.txt`

Expected runtime: ~10–30 min (8 CPUs) depending on cluster load.
