# AI Hardware Detection and Benchmarking Tool

A comprehensive Python tool to detect hardware accelerators on Linux systems and benchmark their performance. This tool checks for GPUs, NPUs, storage accelerators, network accelerators, cryptographic acceleration, video acceleration, compression acceleration, and CPU vector extensions.

## Features

- **Hardware Detection**: Comprehensive scanning for all types of accelerators
- **Performance Benchmarking**: Test CPU and GPU performance with NumPy and Pandas
- **Student-Friendly Output**: Clear explanations of what each accelerator does
- **Cross-Platform**: Works on any Linux system
- **Flexible Testing**: Run full scans, CPU benchmarks, or GPU acceleration tests

## Quick Start

### Option 1: Using the setup script (recommended)

```bash
./setup.sh
```

### Option 2: Manual setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic hardware detection only

```bash
./ai_hw_probe.py
```

### Hardware detection + CPU benchmarks

```bash
./ai_hw_probe.py --benchmark
```

### Hardware detection + CPU and GPU benchmarks

```bash
./ai_hw_probe.py --benchmark --gpu-benchmark
```

Note: `--gpu-benchmark` automatically implies `--benchmark`, so you can use just:

```bash
./ai_hw_probe.py --gpu-benchmark
```

## Command Line Options

| Option | Description |
|---------|-------------|
| `--benchmark` | Run CPU performance benchmarks |
| `--gpu-benchmark` | Run GPU acceleration benchmarks (includes CPU benchmarks) |

## Hardware Detection

The tool scans for and reports on:

### CPU/Processor
- Model name and vendor
- Architecture type
- Device tree information (for embedded systems)

### CPU Speed-Up Features
- **AVX/AVX2/AVX-512**: Advanced Vector Extensions for faster math
- **SSE/SSE2/SSE3/SSE4**: Streaming SIMD Extensions
- **ARM NEON**: SIMD instruction set for ARM processors
- **AES-NI**: Hardware encryption acceleration

### Graphics Processing Unit (GPU)
- Dedicated graphics cards (NVIDIA, AMD, Intel)
- Integrated graphics
- DRM devices and drivers
- Graphics libraries (EGL, GLES, OpenGL)

### Neural Processing Unit (NPU) / AI Chips
- Google Coral TPU
- Neural network accelerators
- Ethos, VSI, NVDLA accelerators

### OpenCL Support
- OpenCL platforms and devices
- General-purpose GPU computing support

### Storage Acceleration
- **NVMe**: Very fast solid-state storage
- **SSD**: Solid-state drives (faster than HDD)
- **RAID**: Multiple drives working together
- **Hardware Encryption**: Dedicated crypto acceleration for storage

### Network Acceleration
- **Network Offload**: Network card handles processing (reduces CPU load)
- **RDMA**: Direct memory access for ultra-fast networking
- **DPDK**: Userspace networking for maximum speed

### Cryptographic Acceleration
- Hardware crypto engines (Intel QAT, CAAM, etc.)
- Dedicated cryptographic processing units

### Video Acceleration
- V4L2 video devices
- Hardware codec support (H.264, H.265, VP9, AV1)

### Compression Acceleration
- Hardware compression (zlib, deflate, lz4, zstd)

### PCIe/USB Accelerator Devices
- Add-on cards that can speed up specific workloads

## Benchmarking

### What the Benchmarks Test

**NumPy Benchmarks (CPU)**
- **Matrix Multiplication (1000×1000)**: Core mathematical operation for AI and physics
- **Element-wise Operations (1M elements)**: Simple math on large arrays
- **Linear Algebra (Eigenvalues)**: Complex matrix operations
- **FFT (1M points)**: Fast Fourier Transform for signal processing

**Pandas Benchmarks (CPU)**
- **DataFrame Creation (100K rows, 10 cols)**: Building data tables
- **DataFrame Filtering (100K rows)**: Selecting data from tables
- **DataFrame GroupBy (100K rows)**: Aggregating data
- **DataFrame Merge (50K rows × 2)**: Combining data tables

**GPU Benchmarks (if GPU available)**
- Matrix multiplication and element-wise operations using:
  - **CuPy**: For NVIDIA GPUs
  - **OpenCL**: For other GPU types

### Understanding Benchmark Results

The benchmarks measure:
- **Time**: How long each test takes (lower is better)
- **Operations/Second**: How many calculations can be done per second (higher is better)
- **Speedup**: How much faster GPU is compared to CPU

### What the Results Mean

**For CPU Benchmarks:**
- Good baseline for comparison
- Shows what your system can do without special hardware
- All modern computers will complete these tests

**For GPU Benchmarks:**
- Tests require compatible GPU and libraries
- Speedup depends on:
  - GPU type (discrete vs integrated)
  - Problem size (larger = bigger GPU advantage)
  - Memory bandwidth
- **Speedup > 5x**: Excellent GPU acceleration
- **Speedup > 2x**: Noticeable improvement
- **Speedup < 2x**: Modest improvement (common for small problems)

## GPU Acceleration Requirements

To run GPU benchmarks, you need:

### For NVIDIA GPUs:
```bash
pip install cupy
```

### For Other GPUs (AMD, Intel, etc.):
```bash
pip install pyopencl
```

**Note**: GPU acceleration is optional. The tool works perfectly without it - you'll just run CPU-only benchmarks.

## Installation of Optional Dependencies

### All features enabled:
```bash
pip install numpy pandas pyopencl cupy
```

### CPU benchmarks only:
```bash
pip install numpy pandas
```

### Hardware detection only (no benchmarks):
```bash
pip install psutil
```

## Example Output

```
============================================================
HARDWARE ACCELERATION CHECK
============================================================

This tool checks your computer for special hardware that can
speed up tasks like graphics, AI, and scientific computing.

------------------------------------------------------------
CPU / PROCESSOR
------------------------------------------------------------
model name          : Intel(R) Core(TM) i7-10700K
vendor_id           : GenuineIntel

------------------------------------------------------------
CPU SPEED-UP FEATURES
------------------------------------------------------------
Your CPU has these special instructions for faster math:
  AVX2
  AVX
  SSE4.2
  AES-NI

[... more hardware detection output ...]

============================================================
PERFORMANCE BENCHMARKS
============================================================

This section tests how fast your computer can do math and
data processing tasks. We measure how many operations per
second your computer can handle.

------------------------------------------------------------
NUMPY (CPU) - MATH AND NUMBER CRUNCHING
------------------------------------------------------------
NumPy is a library for fast number crunching in Python.
Think of it like a super-powered calculator.

Total time for all tests: 2.456 seconds

Tests performed:
  Matrix Multiplication (1000x1000)
    Time: 1.234 seconds
    Speed: 8100000000 operations/second

[... more benchmark results ...]

============================================================
SUMMARY
============================================================

Your system has these acceleration features:
  CPU Vector Extensions
  Graphics (GPU)

This means some programs will run faster than on a basic
computer without these features.
```

## How to Interpret Results

### Hardware Detection
- **Detected features**: Great! Your system can accelerate these types of workloads
- **No features detected**: Normal - you'll use CPU for everything (still works fine)
- **Partial detection**: Some acceleration available for specific tasks

### Benchmarking
- **Fast times**: Your system is well-suited for these operations
- **Slow times**: Consider using hardware acceleration or upgrading components
- **GPU speedup**: If >2x, GPU is helping significantly

## Use Cases

### When to Use This Tool

1. **Before Installing AI Software**: Check if your system supports GPU acceleration
2. **Performance Debugging**: Understand why certain tasks are slow
3. **System Upgrades**: Compare hardware before and after upgrades
4. **Learning**: Understand what hardware accelerators exist and what they do
5. **Cloud Computing**: Check what's available on cloud instances

### When GPU Acceleration Helps Most

- **Machine Learning**: Training neural networks (can be 10-100x faster)
- **Deep Learning**: Inference on large models (5-50x faster)
- **Scientific Computing**: Simulations and modeling (2-20x faster)
- **Video Processing**: Encoding/decoding (3-10x faster)
- **Cryptocurrency Mining**: Hash calculations (100-1000x faster)

### When CPU is Sufficient

- **Small datasets** (<10,000 items)
- **Simple operations** (basic math, string processing)
- **Development and testing**
- **General programming**
- **Light data analysis**

## Troubleshooting

### Benchmark not running?
- Ensure NumPy and Pandas are installed: `pip install numpy pandas`
- Check Python version: Needs Python 3.7+

### GPU benchmarks not running?
- Install GPU libraries: `pip install cupy` (NVIDIA) or `pip install pyopencl` (others)
- Check GPU is recognized: Look at "GRAPHICS PROCESSING UNIT" section
- Some integrated GPUs don't support needed features

### Permission errors?
- The tool needs read access to `/sys/` and `/proc/`
- Run with standard user permissions (no sudo needed)

### Very slow benchmarks?
- Check if other programs are using CPU/GPU heavily
- Close web browsers, games, or other heavy applications
- Some tests can take 10-30 seconds on slower systems

## Architecture Support

- **x86_64** (Intel, AMD): Full support
- **ARM64** (Raspberry Pi, embedded): Full support
- **Other Linux**: Hardware detection works, benchmarks may vary

## Files

- `ai_hw_probe.py`: Main Python script
- `ai_hw_probe.sh`: Original bash script (for reference)
- `requirements.txt`: Python dependencies
- `setup.sh`: Automated setup script
- `README.md`: This documentation

## Contributing

This tool was originally created for NXP i.MX8M Mini embedded systems and has been expanded for generic Linux systems. Contributions and suggestions are welcome!

## License

Original script by Ed Thompson

## Support

For issues, questions, or suggestions, please check the GitHub repository or documentation.

---

**Remember**: Even without special hardware acceleration, your computer is still capable! The benchmarks help you understand what you have and make informed decisions about software and hardware choices.

