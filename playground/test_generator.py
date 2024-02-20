import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
_ = parser.add_argument(
    "--symbols",
    type=int,
    default=1024,
    help="Numer of symbols to generate",
)
_ = parser.add_argument(
    "--samples-per-symbol",
    type=int,
    default=128,
    help="Number of samples per symbol. Default is 128.",
)
_ = parser.add_argument(
    "--frequency-bin",
    type=int,
    default=2,
    help="Frequency bin at which a sine is generated. Default is two.",
)

_ = parser.add_argument(
    "--amplitude",
    type=int,
    default=10**6,
    help="Amplitude of the generated signal. Default is 10^6.",
)

_ = parser.add_argument(
    "--plot",
    action="store_true",
)

_ = parser.add_argument(
    "--file",
    type=argparse.FileType("w"),
    default=sys.stdout,
    help="File to write data to. Default is stdout.",
)

nspace = parser.parse_args()
symbol_count = nspace.symbols
k = nspace.frequency_bin
N = nspace.samples_per_symbol
file = nspace.file
amplitude = nspace.amplitude
sample_count = symbol_count * N
data = np.zeros(sample_count, dtype=np.int32)
prototype = np.round(amplitude * np.cos(2 * np.pi * k * np.arange(N) / N)).astype(
    np.int32
)
data = np.stack((prototype,) * symbol_count).flatten()

print(
    f"""
Symbol count: {symbol_count}
N: {N}
k: {k}
""",
    file=sys.stderr,
)

sys.stderr.write("\n\n")
file.buffer.write(data.tobytes())
file.close()
if nspace.plot:
    plt.plot(data)
    plt.show()
