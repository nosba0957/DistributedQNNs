// DGX side of dqnn_qpu.py qpu: streams each circuit's angles to the OPX over
// OPNIC and writes back every shot's measurement bits.  No ML here: Python
// writes the angles file and turns the bits into predictions.
//
//   dqnn_feeder --params P.bin --n_params K --num_qubits N --num_shots S --out B.bin
//
// P.bin: float64, n_circuits x K angles in turns (row order = circuit order).
// B.bin: uint8, n_circuits x S x N bits, appended after every circuit so a
//        crash keeps the circuits already measured.
//
// Adapted from ADC-QAOA/ADC-QAOA/opnic_dgx_opx/dgx_cpu_qaoa.cpp, minus COBYLA.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <opnic/opnic.hpp>
#include <vector>

// Must match dqnn_qpu.py.
#define INCOMING_STREAM_ID 654 // OPX -> DGX  (measurement bits)
#define OUTGOING_STREAM_ID 987 // DGX -> OPX  (angles)
#define NUM_QUBITS_MAX 5       // per-shot stride in MeasurementPacket
#define SHOTS_PER_PACKET 128
#define MAX_PARAMS 256 // angles per circuit; unused tail is sent as 0
#define BUFFER_SIZE 1024 // stream depth; also the cap on --num_shots

struct MeasurementPacket {
  // [s * NUM_QUBITS_MAX + q] = qubit q on shot s of this packet.
  qm::Value<int, SHOTS_PER_PACKET * NUM_QUBITS_MAX> bits;
  QM_DECLARE_PACKET(MeasurementPacket, bits);
};

struct ParamPacket {
  qm::Value<double, MAX_PARAMS> params; // QUA fixed on the OPX: keep |x| < 8
  QM_DECLARE_PACKET(ParamPacket, params);
};

using incoming_stream_t =
    qm::IncomingStream<MeasurementPacket, BUFFER_SIZE, qm::StreamType::CPU>;
using outgoing_stream_t =
    qm::OutgoingStream<ParamPacket, BUFFER_SIZE, qm::StreamType::CPU>;

static int usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s --params P.bin --n_params K --num_qubits N --num_shots S "
          "--out B.bin\n",
          prog);
  return 1;
}

int main(int argc, char **argv) {
  const char *params_path = nullptr, *out_path = nullptr;
  int n_params = 0, num_qubits = 0, num_shots = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--params"))
      params_path = argv[i + 1];
    else if (!std::strcmp(argv[i], "--out"))
      out_path = argv[i + 1];
    else if (!std::strcmp(argv[i], "--n_params"))
      n_params = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--num_qubits"))
      num_qubits = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--num_shots"))
      num_shots = std::atoi(argv[i + 1]);
    else
      return usage(argv[0]);
  }
  if (!params_path || !out_path || n_params < 1 || n_params > MAX_PARAMS ||
      num_qubits < 1 || num_qubits > NUM_QUBITS_MAX || num_shots < 1 ||
      num_shots > BUFFER_SIZE) {
    fprintf(stderr, "need 1<=n_params<=%d, 1<=num_qubits<=%d, 1<=num_shots<=%d\n",
            MAX_PARAMS, NUM_QUBITS_MAX, BUFFER_SIZE);
    return usage(argv[0]);
  }
  // wait_for_packets indexes its cache per batch; the batch must divide the
  // stream depth (same rule as dgx_cpu_qaoa.cpp).
  int num_packets = (num_shots + SHOTS_PER_PACKET - 1) / SHOTS_PER_PACKET;
  if (BUFFER_SIZE % num_packets != 0) {
    fprintf(stderr, "num_shots=%d -> %d packets, which must divide %d; use a "
                    "power of two\n",
            num_shots, num_packets, BUFFER_SIZE);
    return 1;
  }

  FILE *f = std::fopen(params_path, "rb");
  if (!f) {
    perror(params_path);
    return 1;
  }
  std::vector<double> angles;
  double v;
  while (std::fread(&v, sizeof v, 1, f) == 1)
    angles.push_back(v);
  std::fclose(f);
  if (angles.empty() || angles.size() % n_params != 0) {
    fprintf(stderr, "%s holds %zu doubles, not a multiple of n_params=%d\n",
            params_path, angles.size(), n_params);
    return 1;
  }
  size_t n_circuits = angles.size() / n_params;

  FILE *out = std::fopen(out_path, "wb");
  if (!out) {
    perror(out_path);
    return 1;
  }

  auto in_stream = qm::initialize_stream<incoming_stream_t>(INCOMING_STREAM_ID);
  auto out_stream =
      qm::initialize_stream<outgoing_stream_t>(OUTGOING_STREAM_ID);
  // dqnn_qpu.py waits for this line before starting the OPX job.
  fprintf(stderr, "Streams initialized, waiting for OPX1000 (%zu circuits, "
                  "%d angles, %d shots)\n",
          n_circuits, n_params, num_shots);
  qm::sync();

  std::vector<unsigned char> bits(static_cast<size_t>(num_shots) * num_qubits);
  double total_ms = 0.0;
  auto t_start = std::chrono::steady_clock::now();
  for (size_t c = 0; c < n_circuits; c++) {
    ParamPacket pkt;
    for (int k = 0; k < MAX_PARAMS; k++)
      pkt.params[k] = k < n_params ? angles[c * n_params + k] : 0.0;

    auto t_send = std::chrono::steady_clock::now();
    out_stream->send(pkt);
    in_stream->wait_for_packets(num_packets);
    total_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_send)
                    .count();

    for (int p = 0; p < num_packets; p++) {
      MeasurementPacket m;
      in_stream->get(p, m);
      for (int s = 0; s < SHOTS_PER_PACKET && p * SHOTS_PER_PACKET + s < num_shots; s++)
        for (int q = 0; q < num_qubits; q++)
          bits[(p * SHOTS_PER_PACKET + s) * num_qubits + q] =
              m.bits[s * NUM_QUBITS_MAX + q] ? 1 : 0;
    }
    std::fwrite(bits.data(), 1, bits.size(), out);
    std::fflush(out);
    if ((c + 1) % 100 == 0 || c + 1 == n_circuits)
      fprintf(stderr, "circuit %zu/%zu  avg round-trip %.2f ms\n", c + 1,
              n_circuits, total_ms / (c + 1));
  }
  std::fclose(out);
  fprintf(stderr, "done: %zu circuits in %.1f s\n", n_circuits,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start)
              .count());
  return 0;
}
