# Crust_Mantle_LAB_S

## English

Single-layer crust over mantle to 50 km depth, S-wave incidence.

Maps to `benchmark/test1_S_out` (FDFK2D test1, Moho at 35 km, S incident). Grid depth 50 km approximates upper-mantle/LAB-scale imaging; no deeper LAB benchmark is available.

## Benchmark mapping

| Item | Value |
|------|-------|
| Case ID | `test1_S_out` |
| Model input | `cache/test1_S_out/input/` |
| Workflow | `Crust_Mantle_LAB_S_workflow.ipynb` or `run_workflow.py` |

## Outputs

- `vs_model_section.png`
- `wavefield_snapshots.png`
- `wavefield_uz.gif`
- `receiver_waveforms.png`
- `moveout_uz.png`, `moveout_ux.png`
- `rf_section_sac.png`, `rf_single_mid.png`
- `rf_section_1d_id.png`
