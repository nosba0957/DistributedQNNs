# Distributed Quantum Neural Networks via Partitioned Features Encoding
The paper of "Distributed quantum neural networks via partitioned features encoding" can be read at [Quantum Machine Intelligence](https://rdcu.be/dAVb9). Here, we put the codes used in the paper. 

## Examples
All experiments used in our numerical experiments run from `dqnn.py`:

```
uv sync                                                # own .venv, pinned in pyproject.toml
uv run python dqnn.py                                  # run every experiment
uv run python dqnn.py semeion16x16_4qnn --n-qubits 5   # one experiment, 5 qubits per QNN
uv run python dqnn.py --help
```

- mnist28x28_14qnn: classifying MNIST data set (28x28) with 14 QNNs
- semeion16x16_4qnn: classifying Semeion data set (16x16) with 4 qnns
- semeion16x16_8qnn: classifying Semeion data set (16x16) with 8 qnns
- semeion8x8_2qnn: classifying reduced sized  Semeion data set (8x8) with 2 qnns
- semeion8x8_1qnn: classifying reduced sized  Semeion data set (8x8) 1 qnn

The QNN counts above are for the default 8 qubits (7 for MNIST). With `--n-qubits`,
each QNN takes `2 * n_qubits * (n_block - 1)` features and the number of QNNs follows.

## Inference on the 5q chip (OPNIC)
Training stays in simulation; `dqnn_qpu.py` runs a trained model's circuits on the
`as-qpu-5q4c` chip. All QNNs share one circuit, and the measurement basis is one more
angle, so the OPX compiles **one** QUA program; per circuit this DGX streams the angles
over OPNIC (`opnic/dqnn_feeder`) and gets every shot's bits back. Pattern and constants
follow `../ADC-QAOA/ADC-QAOA/opnic_dgx_opx/`.

```
# 1. train a QPU-sized model here (5 qubits, linear CZ chain, shallow) and save it
uv run python dqnn.py semeion8x8_2qnn --n-qubits 5 --depth 2 --linear-cz --save models

# 2. everything else runs in the lab venv (QuAM, Qiskit 2.x)
LAB=~/as_ntu_ncku/.venv/bin/python   # not `uv run --project`: its sync could touch the shared lab venv
$LAB dqnn_qpu.py check   models/semeion8x8_2qnn_q5_d2_lin.npz             # Qiskit circuits == torchquantum
$LAB dqnn_qpu.py aer     models/semeion8x8_2qnn_q5_d2_lin.npz --images 50 # shots, no noise
$LAB dqnn_qpu.py program models/semeion8x8_2qnn_q5_d2_lin.npz             # transpile + QUA, no hardware

# 3. build the DGX feeder once, then run on the chip
cmake -S opnic -B opnic/build && cmake --build opnic/build
$LAB dqnn_qpu.py qpu     models/semeion8x8_2qnn_q5_d2_lin.npz --images 50
```

Each mode prints accuracy next to the simulator's, how often it predicts the same class,
and the correlation of its outputs with the simulator's (≈1 tracks it, ≈0 is noise).
Results (raw bits included) go to `results/`.

- `qpu` opens its own QM with `close_other_machines=False` and records it in
  `~/.my_open_qms.json`, like `opnic_dgx_opx`; never close other groups' QMs.
- It uses OPNIC stream IDs 654/987, the same as `opnic_dgx_opx`: don't run both at once.
- One circuit carries at most 256 angles (`MAX_PARAMS`), so `--depth` ≤ 7 at 5 qubits.
- Angles travel as turns and every `rz` is applied with `frame_rotation_2pi`: QUA `fixed`
  only holds [-8, 8), and the chip's `rz` macro (`frame_rotation(-angle)`, radians) overflows past it.

### First QPU session
Stop at the first step that fails.

1. Health check: rerun `../../experiments-ethernet/CZ_fidelity.py`. The April run showed
   pair (3,4) at ~0 XEB fidelity, and the linear chain needs every pair.
2. One image (8 circuits): `qpu models/..._d1_lin.npz --images 1`. Outputs should correlate
   with the simulator, not sit near 0.
3. Depth sweep: `--images 5` for the `d1`, `d2`, `d5` models; keep the deepest that still correlates.
4. Classification: `--images 50` (or all 319) at that depth, against the simulator accuracy.
## Citation
If you find this repository useful for your research, please consider citing our work:
```
@article{kawase2024distributed,
  title={Distributed quantum neural networks via partitioned features encoding},
  author={Yoshiaki Kawase},
  journal={Quantum Machine Intelligence},
  volume={6},
  number={1},
  pages={15},
  year={2024},
  publisher={Springer}
}
```