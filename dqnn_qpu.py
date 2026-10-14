"""Inference for a `dqnn.py --save` model: exact check, Aer, or the 5q chip over OPNIC.

Runs in the lab venv (QuAM, Qiskit 2.x), not this directory's torchquantum venv:

    LAB=~/as_ntu_ncku/.venv/bin/python   # not `uv run --project`: its sync could touch the shared lab venv
    $LAB dqnn_qpu.py check   models/M.npz              # Qiskit circuits == torchquantum, exactly
    $LAB dqnn_qpu.py aer     models/M.npz --images 50  # same circuits with shots, no noise
    $LAB dqnn_qpu.py program models/M.npz              # transpile + build QUA, no hardware
    $LAB dqnn_qpu.py qpu     models/M.npz --images 50  # OPX + opnic/dqnn_feeder on this DGX

Every QNN of a model is the same circuit with different angles, and the measurement
basis is one more angle (RY(-pi/2) before measuring turns Z into X), so the QPU runs
ONE compiled QUA program. Per circuit the DGX feeder streams the angles (in turns,
applied as frame_rotation_2pi) and receives every shot's bits.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

HERE = Path(__file__).resolve().parent
FEEDER = HERE / 'opnic' / 'build' / 'dqnn_feeder'
# Must match opnic/dqnn_feeder.cpp: per-shot stride, shots per packet, params per packet.
NUM_QUBITS_MAX = 5
SHOTS_PER_PACKET = 128
MAX_PARAMS = 256
STREAM_OUT, STREAM_IN = 654, 987  # OPX -> DGX bits, DGX -> OPX angles
BASIS = {'Z': 0.0, 'X': -math.pi / 2}  # RY angle before measurement; obs order is [X..., Z...]


def load(path, fold):
    z = np.load(path)
    cfg = json.loads(str(z['config']))
    idx = z[f'test_idx_{fold}']
    return cfg, z['x'][idx], z['y'][idx], z[f'weights_{fold}'], z[f'sim_out_{fold}']


def pick(n_test, n_images):
    """Fixed random subset: the data may be sorted by class, so the first N would be one digit."""
    if not n_images or n_images >= n_test:
        return np.arange(n_test)
    return np.sort(np.random.default_rng(0).choice(n_test, n_images, replace=False))


def circuit(cfg, enc, w, basis, measure=True):
    """Same gate sequence as dqnn.QNNsubModel.forward; wire i is qubit i."""
    n, nb, d, linear = cfg['n_qubits'], cfg['n_block'], cfg['depth'], cfg['linear']
    qc = QuantumCircuit(n, n)

    def rot(angles, s):  # sublayer s: RX on even, RY on odd, angle index i + s*n
        for i in range(n):
            (qc.rx if s % 2 == 0 else qc.ry)(angles[i + s*n], i)

    def cz():
        for i in range(n - linear):
            qc.cz(i, (i + 1) % n)

    for blk in range(nb):
        for k in range(d):
            rot(w, 2*(blk*d + k))
            rot(w, 2*(blk*d + k) + 1)
            if blk < nb - 1 or k < d - 1:
                cz()
        if blk < nb - 1:  # data encoding
            rot(enc, 2*blk)
            rot(enc, 2*blk + 1)
            cz()
    if measure:
        for i in range(n):
            qc.ry(basis, i)
        qc.measure(range(n), range(n))
    return qc


def parametric(cfg):
    enc = ParameterVector('enc', cfg['per_qnn'])
    w = ParameterVector('w', 2*cfg['n_qubits']*cfg['depth']*cfg['n_block'])
    basis = Parameter('basis')
    return circuit(cfg, enc, w, basis), [*enc, *w, basis]


def angles(cfg, x, weights):
    """One row per circuit, ordered (image, qnn, basis Z then X): [encoding chunk, weights, basis] in radians."""
    rows = []
    for img in x:
        for q, chunk in enumerate(np.split(img, cfg['n_qnn'])):
            for b in BASIS.values():
                rows.append(np.concatenate([chunk, weights[q], [b]]))
    return np.array(rows)


def outputs(cfg, bits):
    """bits (n_circuits, shots, n_qubits) -> summed [<X_0..>, <Z_0..>] per image, like dqnn's QNN-sum output."""
    m = cfg['n_class'] // 2
    ev = (1 - 2*bits[:, :, :m].astype(float)).mean(axis=1)  # <Z> per measured qubit, (n_circuits, m)
    ev = ev.reshape(-1, cfg['n_qnn'], 2, m)                 # (image, qnn, basis Z/X, qubit)
    return np.concatenate([ev[:, :, 1], ev[:, :, 0]], axis=2).sum(axis=1)


def report(tag, out, y, sim_out):
    acc = (out.argmax(1) == y).mean()
    sim_acc = (sim_out.argmax(1) == y).mean()
    agree = (out.argmax(1) == sim_out.argmax(1)).mean()
    err = np.abs(out - sim_out).mean()
    corr = np.corrcoef(out.ravel(), sim_out.ravel())[0, 1]  # ~1: tracks the simulator; ~0: noise
    print(f"{tag}: accuracy {acc:.3f} (simulator {sim_acc:.3f}, chance 0.100) | "
          f"same prediction as simulator {agree:.3f} | mean |output - simulator| {err:.4f} | "
          f"correlation with simulator {corr:.3f}")


def save(path, mode, **arrays):
    os.makedirs(HERE / 'results', exist_ok=True)
    out = HERE / 'results' / f"{Path(path).stem}_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(out, **arrays)
    print(f"saved {out}")


# ── check / aer ───────────────────────────────────────────────────────────

def check(cfg, x, y, weights, sim_out, n_images):
    """Statevector expectations of our Qiskit circuits vs the torchquantum outputs saved by dqnn.py."""
    from qiskit.quantum_info import SparsePauliOp, Statevector
    n, m = cfg['n_qubits'], cfg['n_class'] // 2
    obs = [SparsePauliOp.from_sparse_list([(p, [i], 1)], n) for p in 'XZ' for i in range(m)]
    sel = pick(len(x), n_images or 20)
    out = np.zeros((len(sel), 2*m))
    for r, img in enumerate(x[sel]):
        for q, chunk in enumerate(np.split(img, cfg['n_qnn'])):
            sv = Statevector(circuit(cfg, chunk, weights[q], 0, measure=False))
            out[r] += [sv.expectation_value(o).real for o in obs]
    diff = np.abs(out - sim_out[sel]).max()
    print(f"check: {len(sel)} images, max |qiskit - torchquantum| = {diff:.2e}")
    assert diff < 1e-4, "Qiskit circuits do not reproduce the torchquantum model"


def aer(cfg, x, y, weights, sim_out, n_images, shots, path):
    from qiskit_aer import AerSimulator
    sel = pick(len(x), n_images)
    qc, params = parametric(cfg)
    rows = angles(cfg, x[sel], weights)
    circs = [qc.assign_parameters(dict(zip(params, r))) for r in rows]
    result = AerSimulator(seed_simulator=0).run(circs, shots=shots, memory=True).result()
    n = cfg['n_qubits']
    bits = np.array([[[int(s[n - 1 - q]) for q in range(n)] for s in result.get_memory(i)] for i in range(len(circs))],
                    dtype=np.uint8)
    out = outputs(cfg, bits)
    report(f"aer, {len(sel)} images x {shots} shots", out, y[sel], sim_out[sel])
    save(path, 'aer', bits=bits, out=out, y=y[sel], sim_out=sim_out[sel], images=sel, angles=rows)


# ── QPU: QUA program + OPNIC feeder ───────────────────────────────────────

def to_turns(rows):
    """Radians -> turns wrapped into [-0.5, 0.5]: QUA fixed only holds [-8, 8)."""
    t = rows / (2*math.pi)
    return t - np.round(t)


def rz_turns(expr, param_vars):
    """-(angle in turns) for one transpiled rz, as a QUA expression over turn-valued params.

    The chip's rz macro is frame_rotation(-angle) in radians, which overflows past +-8 rad;
    frame_rotation_2pi wraps itself. Transpiled angles are linear in the params, so the
    coefficients come from evaluating at 0 and at each unit vector (as opnic_dgx_opx does)."""
    present = expr.parameters
    const = float(expr.bind({p: 0.0 for p in present}))
    t = const / (2*math.pi)
    qua = -(t - round(t))
    for p in present:
        coeff = float(expr.bind({q: float(q == p) for q in present})) - const
        if abs(coeff) > 1e-10:
            qua = qua + (-coeff) * param_vars[p]
    return qua


def transpiled(cfg, machine):
    from qiskit import transpile
    from quam_libs.experiments.qiskit_circuit import create_target
    qc, params = parametric(cfg)
    qc_t = transpile(qc, target=create_target(machine), initial_layout=list(range(cfg['n_qubits'])),
                     optimization_level=3)
    print(f"transpiled: {qc_t.size()} gates, depth {qc_t.depth()}, {dict(qc_t.count_ops())}")
    return qc, qc_t, params


def verify_transpile(cfg, qc, qc_t, params):
    """Bound with random angles, the transpiled circuit must be the same unitary, and every rz
    evaluated the way the QUA program will (turns, wrapped) must match its bound angle mod 2pi."""
    from qiskit.circuit import ParameterExpression
    from qiskit.quantum_info import Operator
    rng = np.random.default_rng(1)
    vals = dict(zip(params, rng.uniform(-math.pi, math.pi, len(params))))
    strip = lambda c: c.remove_final_measurements(inplace=False)
    a, b = Operator(strip(qc.assign_parameters(vals))), Operator(strip(qc_t.assign_parameters(vals)))
    assert a.equiv(b), "transpiled circuit is not equivalent to the logical one"
    turns = dict(zip(params, to_turns(np.array(list(vals.values())))))
    for inst in qc_t.data:
        if inst.operation.name == 'rz' and isinstance(inst.operation.params[0], ParameterExpression):
            expr = inst.operation.params[0]
            want = float(expr.bind({p: vals[p] for p in expr.parameters})) / (2*math.pi)
            got = -rz_turns(expr, turns)
            assert abs((got - want) - round(got - want)) < 1e-9, (expr, got, want)
    measured = {qc_t.find_bit(i.qubits[0]).index: qc_t.find_bit(i.clbits[0]).index
                for i in qc_t.data if i.operation.name == 'measure'}
    assert measured == {q: q for q in range(cfg['n_qubits'])}, f"qubit -> bit mapping changed: {measured}"
    print("transpile check: equivalent unitary, rz turns match, qubit i -> bit i")


def build_program(cfg, machine, qc_t, params, n_circuits, shots):
    from qm.qua import (Cast, QuaArray, assign, declare, declare_input_stream, declare_output_stream,
                        declare_struct, fixed, for_, program, qua_struct, receive_from_stream, send_to_stream)
    from quam_libs.experiments.qiskit_circuit.qiskit_to_qua import (ensure_resets_for_active_qubits,
                                                                     has_reset_at_boundary)
    n_params = len(params)
    assert n_params <= MAX_PARAMS, f"{n_params} angles per circuit > MAX_PARAMS={MAX_PARAMS}; lower --depth"
    for qubit in machine.active_qubits[:cfg['n_qubits']]:
        qubit.macros['reset'].reset_type = 'active'
        qubit.macros['reset'].max_attempts = 1  # as opnic_dgx_opx: retries re-roll noise until thresholds are recalibrated
    if not has_reset_at_boundary(qc_t):
        qc_t = ensure_resets_for_active_qubits(qc_t)
    creg = qc_t.cregs[0]
    packet_ints = SHOTS_PER_PACKET * NUM_QUBITS_MAX
    num_packets = (shots + SHOTS_PER_PACKET - 1) // SHOTS_PER_PACKET

    @qua_struct
    class ParamPacket:
        params: QuaArray[float, MAX_PARAMS]

    @qua_struct
    class MeasurementPacket:
        bits: QuaArray[int, SHOTS_PER_PACKET * NUM_QUBITS_MAX]

    with program() as prog:
        p_vars = [declare(fixed) for _ in range(n_params)]
        param_struct = declare_struct(ParamPacket)
        param_stream = declare_input_stream("opnic", STREAM_IN, ParamPacket)
        out_struct = declare_struct(MeasurementPacket)
        out_stream = declare_output_stream("opnic", STREAM_OUT, MeasurementPacket)
        circ, shot, pkt, copy_idx = declare(int), declare(int), declare(int), declare(int)
        meas_buffer = declare(int, size=packet_ints * num_packets)
        param_vars = dict(zip(params, p_vars))
        machine.apply_all_flux_to_joint_idle()

        with for_(circ, 0, circ < n_circuits, circ + 1):
            receive_from_stream(param_stream, target_variable=param_struct)
            for k in range(n_params):
                assign(p_vars[k], param_struct.params[k])
            with for_(shot, 0, shot < shots, shot + 1):
                bits = gates(qc_t, machine, param_vars)
                for b in range(creg.size):
                    assign(meas_buffer[shot * NUM_QUBITS_MAX + b], Cast.to_int(bits[b]))
            with for_(pkt, 0, pkt < num_packets, pkt + 1):
                with for_(copy_idx, 0, copy_idx < packet_ints, copy_idx + 1):
                    assign(out_struct.bits[copy_idx], meas_buffer[pkt * packet_ints + copy_idx])
                send_to_stream(out_stream, out_struct)
    return prog


def gates(qc_t, machine, param_vars):
    """opnic_dgx_opx's qiskit_to_qua_macro_parametric, with every rz as frame_rotation_2pi."""
    from qiskit.circuit import ParameterExpression
    from qm.qua import assign, declare
    index = {q: qc_t.find_bit(q).index for q in qc_t.qubits}
    bits = declare(bool, value=[False] * qc_t.num_clbits)
    for inst in qc_t.data:
        op, qubits = inst.operation, inst.qubits
        if len(qubits) == 2:
            (machine.active_qubits[index[qubits[0]]] @ machine.active_qubits[index[qubits[1]]]).apply(op.name, *op.params)
            continue
        qubit = machine.active_qubits[index[qubits[0]]]
        crosses = op.name in ('measure', 'reset')  # xy -> resonator; rz/sx stay on xy, CZ aligns itself
        if crosses:
            qubit.align()
        if op.name == 'rz':
            angle = op.params[0]
            if isinstance(angle, ParameterExpression):
                qubit.xy.frame_rotation_2pi(rz_turns(angle, param_vars))
            else:
                t = float(angle) / (2*math.pi)
                qubit.xy.frame_rotation_2pi(-(t - round(t)))
        else:
            result = qubit.apply(op.name, *op.params)
            for clbit in inst.clbits:
                assign(bits[qc_t.find_bit(clbit).index], result)
        if crosses:
            qubit.align()
    return bits


def lab():
    """QuAM machine for the 5q chip, plus opnic_dgx_opx's QM ledger (never closes other groups' QMs)."""
    os.environ.setdefault('QUAM_STATE_PATH', str(Path.home() / 'as_ntu_ncku' / 'as-qpu-5q4c'))
    sys.path.insert(0, str(HERE.parent / 'ADC-QAOA' / 'ADC-QAOA'))
    import my_qms
    from quam_libs.components import QuAM
    return QuAM.load(), my_qms


def program_only(cfg, path, out_file):
    from qm import generate_qua_script
    machine, _ = lab()
    qc, qc_t, params = transpiled(cfg, machine)
    verify_transpile(cfg, qc, qc_t, params)
    prog = build_program(cfg, machine, qc_t, params, n_circuits=1, shots=1024)
    out_file = out_file or HERE / 'results' / f"{Path(path).stem}.qua.py"
    os.makedirs(Path(out_file).parent, exist_ok=True)
    Path(out_file).write_text(generate_qua_script(prog))
    print(f"QUA program ({len(params)} angles per circuit) written to {out_file}")


def qpu(cfg, x, y, weights, sim_out, n_images, shots, path):
    from qm import QuantumMachinesManager
    num_packets = (shots + SHOTS_PER_PACKET - 1) // SHOTS_PER_PACKET
    assert 1024 % num_packets == 0 and shots <= 1024, "shots must be a power of two <= 1024 (feeder buffer)"
    assert cfg['n_qubits'] <= NUM_QUBITS_MAX
    if not FEEDER.exists():
        sys.exit(f"build the DGX feeder first: cmake -S opnic -B opnic/build && cmake --build opnic/build ({FEEDER})")
    machine, my_qms = lab()
    sel = pick(len(x), n_images)
    rows = angles(cfg, x[sel], weights)
    qc, qc_t, params = transpiled(cfg, machine)
    verify_transpile(cfg, qc, qc_t, params)
    prog = build_program(cfg, machine, qc_t, params, len(rows), shots)

    stamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(HERE / 'results', exist_ok=True)
    params_file = HERE / 'results' / f'params_{stamp}.bin'
    bits_file = HERE / 'results' / f'bits_{stamp}.bin'
    to_turns(rows).astype(np.float64).tofile(params_file)
    print(f"{len(sel)} images -> {len(rows)} circuits x {shots} shots, {len(params)} angles each")

    feeder = subprocess.Popen([str(FEEDER), '--params', str(params_file), '--n_params', str(len(params)),
                               '--num_qubits', str(cfg['n_qubits']), '--num_shots', str(shots),
                               '--out', str(bits_file)], stderr=subprocess.PIPE, text=True)
    for line in feeder.stderr:  # start the OPX job only once the DGX streams are listening
        print(f"[feeder] {line}", end='')
        if 'Streams initialized' in line:
            break
    else:
        sys.exit(f"feeder exited before its streams came up (code {feeder.wait()})")

    settings = dict(host=machine.network['host'], cluster_name=machine.network['cluster_name'],
                    octave=machine.get_octave_config(), timeout=1200)
    if 'port' in machine.network:
        settings['port'] = machine.network['port']
    qm = my_qms.remember(QuantumMachinesManager(**settings).open_qm(machine.generate_config(),
                                                                    close_other_machines=False))
    try:
        job = qm.execute(prog)
        print(f"running on QM {qm.id}, job {job.id}")
        for line in feeder.stderr:
            print(f"[feeder] {line}", end='')
        if feeder.wait() != 0:
            sys.exit(f"feeder failed (code {feeder.returncode}); partial bits in {bits_file}")
        job.wait_until(['Done', 'Canceled', 'Error'])
    finally:
        feeder.kill()
        qm.close()
        my_qms.forget(qm.id)

    bits = np.fromfile(bits_file, dtype=np.uint8).reshape(len(rows), shots, cfg['n_qubits'])
    out = outputs(cfg, bits)
    report(f"qpu, {len(sel)} images x {shots} shots", out, y[sel], sim_out[sel])
    save(path, 'qpu', bits=bits, out=out, y=y[sel], sim_out=sim_out[sel], images=sel, angles=rows,
         state=Path(os.environ['QUAM_STATE_PATH'], 'state.json').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['check', 'aer', 'program', 'qpu'])
    p.add_argument('model', help='.npz written by dqnn.py --save')
    p.add_argument('--fold', type=int, default=0, help='which fold\'s weights and test images (default 0)')
    p.add_argument('--images', type=int, help='random subset of the test images (default: all; check: 20)')
    p.add_argument('--shots', type=int, default=1024)
    p.add_argument('--output', help='program: where to write the QUA script')
    a = p.parse_args()
    cfg, x, y, weights, sim_out = load(a.model, a.fold)
    if a.mode == 'check':
        check(cfg, x, y, weights, sim_out, a.images)
    elif a.mode == 'aer':
        aer(cfg, x, y, weights, sim_out, a.images, a.shots, a.model)
    elif a.mode == 'program':
        program_only(cfg, a.model, a.output)
    else:
        qpu(cfg, x, y, weights, sim_out, a.images, a.shots, a.model)
