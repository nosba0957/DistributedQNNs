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