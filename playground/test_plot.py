import sys
import numpy as np

buffer = bytearray()
for line in sys.stdin.buffer.readlines():
    buffer += line
data = np.frombuffer(buffer, dtype=np.complex128)
print(data)
print(np.abs(data))
