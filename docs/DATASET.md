# NASA C-MAPSS Dataset Specification (FD001)

## 1. Provenance & Attribution
- **Dataset Title**: Turbofan Engine Degradation Simulation Data Set
- **Publisher**: NASA Prognostics Center of Excellence (PCoE) & NASA Open Data Portal
- **Authors**: A. Saxena, K. Goebel, D. Simon, N. Eklund (2008)
- **Sub-dataset Used**: **FD001** (Single operating condition at Sea Level, High-Pressure Compressor failure mode).

---

## 2. File Manifest & Structure

| File Name | Row Count | Column Count | Description |
| :--- | :---: | :---: | :--- |
| `train_FD001.txt` | 20,631 | 26 | Complete run-to-failure operational cycles for 100 engines |
| `test_FD001.txt` | 13,096 | 26 | Truncated operational trajectories for 100 test engines |
| `RUL_FD001.txt` | 100 | 1 | Official ground truth Remaining Useful Life at last test cycle |

### Column Layout (26 Variables)
1. `unit_id`: Engine identifier (integer $1 \dots 100$)
2. `cycle`: Operational cycle index ($1 \dots C_{\text{final}}$)
3. `setting_1`: Operational setting 1 (Mach number)
4. `setting_2`: Operational setting 2 (Altitude / throttle resolver angle)
5. `setting_3`: Operational setting 3 (Sea-level condition constant)
6. `sensor_1` through `sensor_21`: Anonymized aerothermal sensor telemetry (temperatures, pressures, rotor speeds, fuel flow, bleed enthalpy).

---

## 3. Feature Selection & Variance Analysis

In FD001 (sea-level single condition), 7 channels have zero or near-zero variance ($\sigma^2 < 10^{-12}$) and were dropped:
- `setting_3` ($\sigma^2 = 0$)
- `sensor_1` ($\sigma^2 = 0$)
- `sensor_5` ($\sigma^2 = 2.8 \times 10^{-29}$)
- `sensor_10` ($\sigma^2 = 0$)
- `sensor_16` ($\sigma^2 = 1.2 \times 10^{-35}$)
- `sensor_18` ($\sigma^2 = 0$)
- `sensor_19` ($\sigma^2 = 0$)

### 17 Retained Features (15 Sensors + 2 Settings)
`setting_1`, `setting_2`, `sensor_2`, `sensor_3`, `sensor_4`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`.
