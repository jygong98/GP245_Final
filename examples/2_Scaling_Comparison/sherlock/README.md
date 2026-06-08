# Python FDFK scaling benchmarks on Sherlock

Run each scaling case as an **individual SLURM job** on Stanford Sherlock (or Oak), then merge per-job JSON outputs into `python_scaling_times.json`.

## Benchmarks

| Suite | Labels | Grid / threads |
|-------|--------|----------------|
| Thread | `python_t1` … `python_t12` | 1001×251, 1–12 threads, nt=4001 (tmax=40 s) |
| Spatial | `python_s1`, `python_s2`, `python_s4`, `python_s8`, `python_s16` | Fortran-matched grids, 8 threads, nt=4001 |

Input data: `benchmark/test1/input` (read by `run_python_scaling.py`).

## Prerequisites

1. Sync the project to Sherlock/Oak (see `doc/sherlock_moho_state_comparison_checklist.md` for rsync examples).
2. Create or copy the **GP245_Final** conda environment on the cluster.
3. Edit `job.sbatch` if needed:
   - `FDFK_PYTHON` default path
   - `PYTHON` or the `module load` / `conda activate` block
   - `#SBATCH --account=YOUR_ACCOUNT` (uncomment and set)
   - Optional email notifications

## Quick start

```bash
cd examples/2_Scaling_Comparison/sherlock

# Optional: override project path and SLURM account
export FDFK_PYTHON=/oak/stanford/groups/sklemp/jygong/GP245/FDFK_Python
export SLURM_ACCOUNT=your_sponsor_account

bash submit_all.sh

# After all jobs complete:
bash merge_results.sh
```

Merged output: `examples/2_Scaling_Comparison/python_scaling_times.json`

## Files

| File | Purpose |
|------|---------|
| `job.sbatch` | Parameterized template; one simulation per job via `--worker` |
| `submit_all.sh` | Submits 17 jobs (12 thread + 5 spatial) with resource limits |
| `merge_results.py` | Python merge logic |
| `merge_results.sh` | Shell wrapper for merge |
| `results/` | Per-job JSON (`python_t8.json`, etc.) — no concurrent writes |
| `logs/` | SLURM stdout/stderr per job |

## Per-job output and merge workflow

Each job runs:

```bash
run_python_scaling.py --worker ... --output-json results/<LABEL>.json
```

`--output-json` writes a **private** file per job, avoiding races on a shared cache. After all jobs finish, `merge_results.sh` collects `results/*.json` and writes the combined `python_scaling_times.json` (one run per label; later files win if a label is duplicated).

## SLURM defaults (adjust in `submit_all.sh`)

| Job group | CPUs | Time | Memory |
|-----------|------|------|--------|
| Thread `python_t1`–`t12` | 1–12 (matches threads) | 4 h | 32G |
| Spatial `python_s1`–`s4` | 8 | 4–6 h | 32–48G |
| Spatial `python_s8` | 8 | 8 h | 64G |
| Spatial `python_s16` | 8 | 12 h | 128G |

Partition: `normal` (override with `SLURM_PARTITION=owners`).

## Example manual submissions

Thread scaling (8 threads):

```bash
sbatch --job-name=python_t8 --cpus-per-task=8 --time=04:00:00 --mem=32G \
  --output=logs/python_t8-%j.out --error=logs/python_t8-%j.err \
  --export=ALL,LABEL=python_t8,SUITE=thread,THREADS=8,NX=1001,NZ=251,NT=4001 \
  job.sbatch
```

Spatial scaling (scale 4):

```bash
sbatch --job-name=python_s4 --cpus-per-task=8 --time=06:00:00 --mem=48G \
  --output=logs/python_s4-%j.out --error=logs/python_s4-%j.err \
  --export=ALL,LABEL=python_s4,SUITE=spatial,THREADS=8,NX=2001,NZ=501,NT=4001,SPATIAL_FACTOR=4 \
  job.sbatch
```

## Monitoring

```bash
squeue -u $USER
ls results/
tail -f logs/python_t8-*.out
```

If a job fails, re-submit only that case (same `sbatch` line from `submit_all.sh` or the examples above), then re-run `merge_results.sh`.
