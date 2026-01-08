#!/usr/bin/env python3

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


class HardwareDetector:
    def __init__(self):
        self.dmesg = self._get_dmesg()
        self.libpaths = ["/usr/lib", "/usr/local/lib", "/lib"]

    def _get_dmesg(self) -> str:
        try:
            result = subprocess.run(
                ["dmesg"], capture_output=True, text=True, timeout=5
            )
            return result.stdout
        except Exception:
            return ""

    def _read_file(self, path: str) -> Optional[str]:
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception:
            return None

    def _file_exists(self, path: str) -> bool:
        return os.path.exists(path)

    def _command_available(self, cmd: str) -> bool:
        try:
            subprocess.run(["which", cmd], capture_output=True, check=True)
            return True
        except Exception:
            return False

    def _run_command(self, cmd: List[str]) -> Tuple[int, str]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return result.returncode, result.stdout
        except Exception as e:
            return -1, str(e)

    def detect_cpu(self) -> Dict[str, str]:
        info: Dict[str, str] = {}
        cpuinfo = self._read_file("/proc/cpuinfo")
        if cpuinfo:
            for line in cpuinfo.split("\n"):
                for key in [
                    "Hardware",
                    "Model",
                    "processor",
                    "CPU architecture",
                    "CPU part",
                    "CPU implementer",
                    "model name",
                    "vendor_id",
                ]:
                    if key in line:
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            info[key] = parts[1].strip()

        compatible = self._read_file("/proc/device-tree/compatible")
        if compatible:
            info["device_tree"] = compatible.replace("\x00", " ").strip()

        model = self._read_file("/sys/firmware/devicetree/base/model")
        if model:
            info["devicetree_model"] = model.replace("\x00", "").strip()

        return info

    def detect_gpu(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "present": False,
            "devices": [],
            "modules": [],
            "libraries": [],
            "details": [],
        }

        gpu_keywords = [
            "vivante",
            "galcore",
            "etnaviv",
            "imx-gpu",
            "gc[0-9]+",
            "mxc_g2d",
            "g2d",
            "pxp",
            "imx-pxp",
            "drm",
            "etnaviv",
            "vsi",
        ]

        if self.dmesg:
            for keyword in gpu_keywords:
                if re.search(keyword, self.dmesg, re.IGNORECASE) is not None:
                    results["present"] = True
                    results["details"].append(f"dmesg: {keyword} found")

        device_nodes = [
            "/dev/dri",
            "/dev/galcore",
            "/dev/mxc_g2d",
            "/dev/pxp",
            "/dev/vsi",
        ]
        for node in device_nodes:
            if self._file_exists(node):
                results["present"] = True
                results["devices"].append(node)

        if self._file_exists("/dev/dri"):
            try:
                devices = os.listdir("/dev/dri")
                for dev in devices:
                    results["devices"].append(f"/dev/dri/{dev}")
            except Exception:
                pass

        if self._command_available("lsmod"):
            code, output = self._run_command(["lsmod"])
            if code == 0:
                for keyword in [
                    "etnaviv",
                    "galcore",
                    "imx_gpu",
                    "mxc_g2d",
                    "pxp",
                    "drm",
                    "gpu",
                    "vsi",
                ]:
                    if re.search(keyword, output, re.IGNORECASE) is not None:
                        results["modules"].append(keyword)

        for libpath in self.libpaths:
            if self._file_exists(libpath):
                try:
                    libs = os.listdir(libpath)
                    for lib in libs:
                        if (
                            re.search(r"libEGL|libGLES|libGL", lib, re.IGNORECASE)
                            is not None
                        ):
                            results["libraries"].append(os.path.join(libpath, lib))
                except Exception:
                    pass

        return results

    def detect_npu(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "present": False,
            "devices": [],
            "modules": [],
            "libraries": [],
            "details": [],
        }

        npu_keywords = ["npu", "ethos", "vsi", "tpu", "neural", "nvdla", "openvit"]

        if self.dmesg:
            for keyword in npu_keywords:
                if re.search(keyword, self.dmesg, re.IGNORECASE) is not None:
                    results["present"] = True
                    results["details"].append(f"dmesg: {keyword} found")

        for libpath in self.libpaths:
            if self._file_exists(libpath):
                try:
                    libs = os.listdir(libpath)
                    for lib in libs:
                        if (
                            re.search(
                                r"libvsi|libethosu|libtpu|libnvdla", lib, re.IGNORECASE
                            )
                            is not None
                        ):
                            results["present"] = True
                            results["libraries"].append(os.path.join(libpath, lib))
                except Exception:
                    pass

        if self._command_available("lsmod"):
            code, output = self._run_command(["lsmod"])
            if code == 0:
                for keyword in ["npu", "ethos", "vsi", "tpu", "nvdla"]:
                    if re.search(keyword, output, re.IGNORECASE) is not None:
                        results["present"] = True
                        results["modules"].append(keyword)

        return results

    def detect_opencl(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"available": False, "platforms": []}

        if self._command_available("clinfo"):
            code, output = self._run_command(["clinfo"])
            if code == 0 and output:
                results["available"] = True
                results["platforms"].append(output[:1000])

        for libpath in self.libpaths:
            if self._file_exists(libpath):
                try:
                    libs = os.listdir(libpath)
                    for lib in libs:
                        if "OpenCL" in lib:
                            results["available"] = True
                            results["platforms"].append(f"Found library: {lib}")
                except Exception:
                    pass

        return results

    def detect_pci_devices(self) -> List[str]:
        devices: List[str] = []
        if not self._command_available("lspci"):
            return devices

        code, output = self._run_command(["lspci", "-nn"])
        if code == 0:
            for line in output.split("\n"):
                if (
                    re.search(
                        r"VGA|3D|Display|NVIDIA|AMD|GPU|Intel.*Graphics|Tesla|Quadro|FirePro|Radeon",
                        line,
                        re.IGNORECASE,
                    )
                    is not None
                ):
                    devices.append(line.strip())
        return devices

    def detect_usb_devices(self) -> List[str]:
        devices: List[str] = []
        if not self._command_available("lsusb"):
            return devices

        code, output = self._run_command(["lsusb"])
        if code == 0:
            for line in output.split("\n"):
                if (
                    re.search(
                        r"Google|Coral|GlobalUnichip|NVIDIA|AMD|Intel.*Graphics|TPU",
                        line,
                        re.IGNORECASE,
                    )
                    is not None
                ):
                    devices.append(line.strip())
        return devices

    def detect_cpu_features(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"flags": [], "vector_extensions": []}

        cpuinfo = self._read_file("/proc/cpuinfo")
        if cpuinfo:
            flags_match = re.search(r"flags\s*:\s*(.+)", cpuinfo, re.IGNORECASE)
            if flags_match:
                flags = flags_match.group(1).split()
                results["flags"] = flags[:20]

                vector_exts = {
                    "avx512": "AVX-512",
                    "avx2": "AVX2",
                    "avx": "AVX",
                    "neon": "ARM NEON",
                    "sse4_2": "SSE4.2",
                    "sse4_1": "SSE4.1",
                    "sse3": "SSE3",
                    "aes": "AES-NI",
                }
                for flag, name in vector_exts.items():
                    if flag in flags:
                        results["vector_extensions"].append(name)

        return results

    def detect_drm_devices(self) -> List[Dict[str, str]]:
        devices: List[Dict[str, str]] = []
        drm_path = "/sys/class/drm"
        if not self._file_exists(drm_path):
            return devices

        try:
            for card in Path(drm_path).glob("card*"):
                if card.is_dir():
                    dev_info: Dict[str, str] = {"name": card.name}
                    for prop in ["modalias", "vendor", "device", "uevent"]:
                        prop_path = card / "device" / prop
                        if prop_path.exists():
                            content = self._read_file(str(prop_path))
                            if content:
                                dev_info[prop] = content.strip()
                    devices.append(dev_info)
        except Exception:
            pass

        return devices

    def detect_storage_accelerators(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "nvme": False,
            "ssd": False,
            "raid": False,
            "encryption": False,
            "details": [],
        }

        nvme_path = "/sys/class/nvme"
        if self._file_exists(nvme_path):
            try:
                nvme_count = len(list(Path(nvme_path).glob("nvme*")))
                if nvme_count > 0:
                    results["nvme"] = True
                    results["details"].append(f"Found {nvme_count} NVMe device(s)")
            except Exception:
                pass

        block_path = "/sys/block"
        if self._file_exists(block_path):
            try:
                for device in Path(block_path).iterdir():
                    queue_path = device / "queue" / "rotational"
                    if queue_path.exists():
                        content = self._read_file(str(queue_path))
                        if content and content.strip() == "0":
                            results["ssd"] = True
                            results["details"].append(
                                f"Non-rotational device: {device.name}"
                            )
            except Exception:
                pass

        if self._command_available("mdadm"):
            code, output = self._run_command(["cat", "/proc/mdstat"])
            if code == 0 and "md" in output:
                results["raid"] = True
                results["details"].append("Software RAID detected")

        if self.dmesg:
            if re.search(r"raid|md\d+|dm-raid", self.dmesg, re.IGNORECASE) is not None:
                results["raid"] = True
                results["details"].append("RAID references in dmesg")

            if (
                re.search(
                    r"crypto|intel.*qat|openssl.*accel|cryptodev",
                    self.dmesg,
                    re.IGNORECASE,
                )
                is not None
            ):
                results["encryption"] = True
                results["details"].append("Hardware crypto acceleration detected")

        return results

    def detect_network_accelerators(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "offload": False,
            "rdma": False,
            "dpdk": False,
            "features": [],
            "details": [],
        }

        net_path = "/sys/class/net"
        if self._file_exists(net_path):
            try:
                for iface in Path(net_path).iterdir():
                    if iface.is_dir() and iface.name not in ["lo"]:
                        features_path = iface / "features"
                        if features_path.exists():
                            content = self._read_file(str(features_path))
                            if content:
                                offload_features = [
                                    "tx",
                                    "rx",
                                    "sg",
                                    "tso",
                                    "gso",
                                    "gro",
                                    "lro",
                                ]
                                for feat in offload_features:
                                    if f"{feat} " in content.lower():
                                        results["offload"] = True
                                        results["features"].append(
                                            f"{iface.name}: {feat}"
                                        )

                        if self._command_available("ethtool"):
                            code, output = self._run_command(
                                ["ethtool", "-k", iface.name]
                            )
                            if code == 0:
                                if "offload" in output.lower():
                                    results["offload"] = True
            except Exception:
                pass

        if self._command_available("ibv_devinfo"):
            code, output = self._run_command(["ibv_devinfo"])
            if code == 0:
                results["rdma"] = True
                results["details"].append("RDMA devices detected")

        rdma_path = "/sys/class/infiniband"
        if self._file_exists(rdma_path):
            try:
                if len(list(Path(rdma_path).iterdir())) > 0:
                    results["rdma"] = True
                    results["details"].append("RDMA devices present")
            except Exception:
                pass

        if self.dmesg:
            if (
                re.search(
                    r"dpdk|vfio|uio_pci_generic|igb_uio", self.dmesg, re.IGNORECASE
                )
                is not None
            ):
                results["dpdk"] = True
                results["details"].append("DPDK references detected")

        return results

    def detect_crypto_accelerators(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"hardware": False, "modules": [], "details": []}

        crypto_modules = [
            "intel_qat",
            "crypto_dev",
            "ccp",
            "nx-crypto",
            "caam",
            "mxc-crypto",
        ]

        if self._command_available("lsmod"):
            code, output = self._run_command(["lsmod"])
            if code == 0:
                for mod in crypto_modules:
                    if re.search(mod, output, re.IGNORECASE) is not None:
                        results["hardware"] = True
                        results["modules"].append(mod)

        crypto_path = "/sys/class/misc"
        if self._file_exists(crypto_path):
            try:
                for dev in Path(crypto_path).iterdir():
                    if "crypto" in dev.name.lower() or "qat" in dev.name.lower():
                        results["hardware"] = True
                        results["details"].append(f"Crypto device: {dev.name}")
            except Exception:
                pass

        return results

    def detect_video_accelerators(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"v4l2": False, "codecs": [], "details": []}

        v4l_path = "/sys/class/video4linux"
        if self._file_exists(v4l_path):
            try:
                devices = list(Path(v4l_path).iterdir())
                if len(devices) > 0:
                    results["v4l2"] = True
                    results["details"].append(f"Found {len(devices)} video device(s)")
            except Exception:
                pass

        if self.dmesg:
            codec_patterns = [
                "vcodec",
                "vpudec",
                "vpudenc",
                "vpuenc",
                "h264",
                "h265",
                "vp9",
                "av1",
            ]
            for pattern in codec_patterns:
                if re.search(pattern, self.dmesg, re.IGNORECASE) is not None:
                    results["codecs"].append(pattern)

        return results

    def detect_compression_accelerators(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"hardware": False, "details": []}

        if self.dmesg:
            if (
                re.search(
                    r"zlib|deflate|lz4|zstd.*accel|qat.*compress",
                    self.dmesg,
                    re.IGNORECASE,
                )
                is not None
            ):
                results["hardware"] = True
                results["details"].append("Hardware compression acceleration detected")

        return results


class BenchmarkRunner:
    def __init__(self):
        self.numpy_available = False
        self.pandas_available = False
        self.pyopencl_available = False
        self.cupy_available = False
        self._check_libraries()

    def _check_libraries(self):
        try:
            import numpy as np

            self.np = np
            self.numpy_available = True
        except ImportError:
            self.np = None

        try:
            import pandas as pd

            self.pd = pd
            self.pandas_available = True
        except ImportError:
            self.pd = None

        try:
            import pyopencl as cl

            self.cl = cl
            self.pyopencl_available = True
        except ImportError:
            self.cl = None

        try:
            import cupy as cp

            self.cp = cp
            self.cupy_available = True
        except ImportError:
            self.cp = None

    def run_numpy_cpu_benchmark(self) -> Dict[str, Any]:
        results = {"available": False, "tests": [], "total_time": 0, "errors": []}

        if not self.numpy_available:
            results["errors"].append("NumPy not installed")
            return results

        results["available"] = True

        try:
            test_data = {
                "matrix_multiply": {
                    "name": "Matrix Multiplication (1000x1000)",
                    "func": self._benchmark_matrix_multiply,
                },
                "elementwise_ops": {
                    "name": "Element-wise Operations (1M elements)",
                    "func": self._benchmark_elementwise_ops,
                },
                "linear_algebra": {
                    "name": "Linear Algebra (Eigenvalues)",
                    "func": self._benchmark_linear_algebra,
                },
                "fft": {
                    "name": "Fast Fourier Transform (1M points)",
                    "func": self._benchmark_fft,
                },
            }

            total_time = 0
            for test_key, test in test_data.items():
                try:
                    runtime, ops_per_sec = test["func"]()
                    total_time += runtime
                    results["tests"].append(
                        {
                            "name": test["name"],
                            "time": runtime,
                            "ops_per_sec": ops_per_sec,
                        }
                    )
                except Exception as e:
                    results["errors"].append(f"{test_key}: {str(e)}")

            results["total_time"] = total_time

        except Exception as e:
            results["errors"].append(f"General error: {str(e)}")

        return results

    def _benchmark_matrix_multiply(self) -> Tuple[float, float]:
        import numpy as np

        size = 1000
        a = np.random.random((size, size))
        b = np.random.random((size, size))

        start = time.time()
        for _ in range(5):
            c = np.dot(a, b)
        end = time.time()

        runtime = end - start
        ops_per_sec = (2 * size**3 * 5) / runtime
        return runtime, ops_per_sec

    def _benchmark_elementwise_ops(self) -> Tuple[float, float]:
        import numpy as np

        size = 1_000_000
        a = np.random.random(size)
        b = np.random.random(size)

        start = time.time()
        for _ in range(10):
            c = a * b + a / b - np.sqrt(a)
        end = time.time()

        runtime = end - start
        ops_per_sec = (4 * size * 10) / runtime
        return runtime, ops_per_sec

    def _benchmark_linear_algebra(self) -> Tuple[float, float]:
        import numpy as np

        size = 500
        a = np.random.random((size, size))

        start = time.time()
        for _ in range(5):
            eigvals = np.linalg.eigvals(a)
        end = time.time()

        runtime = end - start
        return runtime, 0

    def _benchmark_fft(self) -> Tuple[float, float]:
        import numpy as np
        import math

        size = 1_000_000
        a = np.random.random(size)

        start = time.time()
        for _ in range(10):
            fft_result = np.fft.fft(a)
        end = time.time()

        runtime = end - start
        ops_per_sec = (size * math.log2(size) * 10) / runtime
        return runtime, ops_per_sec

    def run_pandas_cpu_benchmark(self) -> Dict[str, Any]:
        results = {"available": False, "tests": [], "total_time": 0, "errors": []}

        if not self.pandas_available:
            results["errors"].append("Pandas not installed")
            return results

        results["available"] = True

        try:
            test_data = {
                "dataframe_creation": {
                    "name": "DataFrame Creation (100K rows, 10 cols)",
                    "func": self._benchmark_dataframe_creation,
                },
                "dataframe_filter": {
                    "name": "DataFrame Filtering (100K rows)",
                    "func": self._benchmark_dataframe_filter,
                },
                "dataframe_groupby": {
                    "name": "DataFrame GroupBy (100K rows)",
                    "func": self._benchmark_dataframe_groupby,
                },
                "dataframe_merge": {
                    "name": "DataFrame Merge (50K rows x2)",
                    "func": self._benchmark_dataframe_merge,
                },
            }

            total_time = 0
            for test_key, test in test_data.items():
                try:
                    runtime, ops_per_sec = test["func"]()
                    total_time += runtime
                    results["tests"].append(
                        {
                            "name": test["name"],
                            "time": runtime,
                            "ops_per_sec": ops_per_sec,
                        }
                    )
                except Exception as e:
                    results["errors"].append(f"{test_key}: {str(e)}")

            results["total_time"] = total_time

        except Exception as e:
            results["errors"].append(f"General error: {str(e)}")

        return results

    def _benchmark_dataframe_creation(self) -> Tuple[float, float]:
        import numpy as np
        import pandas as pd

        rows = 100_000

        data = {f"col_{i}": np.random.random(rows) for i in range(10)}

        start = time.time()
        for _ in range(5):
            df = pd.DataFrame(data)
        end = time.time()

        runtime = end - start
        ops_per_sec = (rows * 10 * 5) / runtime
        return runtime, ops_per_sec

    def _benchmark_dataframe_filter(self) -> Tuple[float, float]:
        import numpy as np
        import pandas as pd

        rows = 100_000

        data = {f"col_{i}": np.random.random(rows) for i in range(3)}
        df = pd.DataFrame(data)

        start = time.time()
        for _ in range(10):
            filtered = df[df["col_0"] > 0.5]
        end = time.time()

        runtime = end - start
        ops_per_sec = (rows * 10) / runtime
        return runtime, ops_per_sec

    def _benchmark_dataframe_groupby(self) -> Tuple[float, float]:
        import numpy as np
        import pandas as pd

        rows = 100_000

        data = {
            "group_col": np.random.randint(0, 100, rows),
            "value_col": np.random.random(rows),
        }
        df = pd.DataFrame(data)

        start = time.time()
        for _ in range(10):
            grouped = df.groupby("group_col")["value_col"].mean()
        end = time.time()

        runtime = end - start
        ops_per_sec = (rows * 10) / runtime
        return runtime, ops_per_sec

    def _benchmark_dataframe_merge(self) -> Tuple[float, float]:
        import numpy as np
        import pandas as pd

        rows = 50_000

        data1 = {"key": np.arange(rows), "value1": np.random.random(rows)}
        data2 = {"key": np.arange(rows), "value2": np.random.random(rows)}
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        start = time.time()
        for _ in range(5):
            merged = pd.merge(df1, df2, on="key")
        end = time.time()

        runtime = end - start
        ops_per_sec = (rows * 5) / runtime
        return runtime, ops_per_sec

    def run_numpy_gpu_benchmark(self) -> Dict[str, Any]:
        results = {
            "available": False,
            "method": None,
            "tests": [],
            "total_time": 0,
            "errors": [],
            "speedup": [],
        }

        if not self.numpy_available:
            results["errors"].append("NumPy not installed")
            return results

        cpu_results = self.run_numpy_cpu_benchmark()
        if not cpu_results["available"]:
            results["errors"].append("CPU benchmark failed")
            return results

        results["available"] = True

        if self.cupy_available:
            results["method"] = "CuPy (NVIDIA GPU)"
            gpu_results = self._run_cupy_benchmark(cpu_results)
        elif self.pyopencl_available:
            results["method"] = "OpenCL"
            gpu_results = self._run_opencl_benchmark(cpu_results)
        else:
            results["errors"].append(
                "No GPU acceleration library available (tried CuPy, OpenCL)"
            )
            return results

        results["tests"] = gpu_results["tests"]
        results["total_time"] = gpu_results["total_time"]
        results["errors"] = gpu_results["errors"]
        results["speedup"] = gpu_results.get("speedup", [])

        return results

    def _run_cupy_benchmark(self, cpu_results: Dict[str, Any]) -> Dict[str, Any]:
        results = {"tests": [], "total_time": 0, "errors": [], "speedup": []}

        try:
            import cupy as cp
            import numpy as np

            test_funcs = {
                "matrix_multiply": lambda: self._benchmark_cupy_matrix_multiply(),
                "elementwise_ops": lambda: self._benchmark_cupy_elementwise_ops(),
            }

            total_time = 0
            for test_key, func in test_funcs.items():
                try:
                    gpu_time, ops_per_sec = func()
                    total_time += gpu_time

                    cpu_test = next(
                        (
                            t
                            for t in cpu_results["tests"]
                            if test_key in t["name"].lower()
                        ),
                        None,
                    )
                    speedup = 0
                    if cpu_test:
                        speedup = cpu_test["time"] / gpu_time

                    results["tests"].append(
                        {
                            "name": f"GPU - {test_key.replace('_', ' ').title()}",
                            "time": gpu_time,
                            "ops_per_sec": ops_per_sec,
                        }
                    )
                    results["speedup"].append({"test": test_key, "speedup": speedup})
                except Exception as e:
                    results["errors"].append(f"{test_key}: {str(e)}")

            results["total_time"] = total_time

        except Exception as e:
            results["errors"].append(f"CuPy benchmark error: {str(e)}")

        return results

    def _benchmark_cupy_matrix_multiply(self) -> Tuple[float, float]:
        import cupy as cp

        size = 1000
        a = cp.random.random((size, size))
        b = cp.random.random((size, size))

        cp.cuda.Stream.null.synchronize()
        start = time.time()
        for _ in range(5):
            c = cp.dot(a, b)
            cp.cuda.Stream.null.synchronize()
        end = time.time()

        runtime = end - start
        ops_per_sec = (2 * size**3 * 5) / runtime
        return runtime, ops_per_sec

    def _benchmark_cupy_elementwise_ops(self) -> Tuple[float, float]:
        import cupy as cp

        size = 1_000_000
        a = cp.random.random(size)
        b = cp.random.random(size)

        cp.cuda.Stream.null.synchronize()
        start = time.time()
        for _ in range(10):
            c = a * b + a / b - cp.sqrt(a)
            cp.cuda.Stream.null.synchronize()
        end = time.time()

        runtime = end - start
        ops_per_sec = (4 * size * 10) / runtime
        return runtime, ops_per_sec

    def _run_opencl_benchmark(self, cpu_results: Dict[str, Any]) -> Dict[str, Any]:
        results = {"tests": [], "total_time": 0, "errors": [], "speedup": []}

        try:
            import pyopencl as cl
            import numpy as np

            platforms = cl.get_platforms()
            if not platforms:
                results["errors"].append("No OpenCL platforms found")
                return results

            ctx = cl.Context(dev_type=cl.device_type.GPU)
            if not ctx.devices:
                results["errors"].append("No GPU devices found for OpenCL")
                return results

            queue = cl.CommandQueue(ctx)

            size = 1000
            a_host = np.random.random((size, size)).astype(np.float32)
            b_host = np.random.random((size, size)).astype(np.float32)

            a_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_host
            )
            b_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_host
            )
            c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, a_host.nbytes)

            kernel = """
            __kernel void matmul(__global const float *A, __global const float *B, __global float *C, int N) {
                int row = get_global_id(0);
                int col = get_global_id(1);
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
            """

            prg = cl.Program(ctx, kernel).build()

            start = time.time()
            for _ in range(5):
                prg.matmul(
                    queue, (size, size), None, a_buf, b_buf, c_buf, np.int32(size)
                )
                queue.finish()
            end = time.time()

            runtime = end - start
            ops_per_sec = (2 * size**3 * 5) / runtime

            results["tests"].append(
                {
                    "name": "GPU - Matrix Multiplication (OpenCL)",
                    "time": runtime,
                    "ops_per_sec": ops_per_sec,
                }
            )

            cpu_test = next(
                (
                    t
                    for t in cpu_results["tests"]
                    if "matrix multiply" in t["name"].lower()
                ),
                None,
            )
            if cpu_test:
                results["speedup"].append(
                    {"test": "matrix_multiply", "speedup": cpu_test["time"] / runtime}
                )

            results["total_time"] = runtime

        except Exception as e:
            results["errors"].append(f"OpenCL benchmark error: {str(e)}")

        return results

    def _benchmark_cupy_matrix_multiply(self) -> Tuple[float, float]:
        import cupy as cp

        size = 1000
        a = cp.random.random((size, size))
        b = cp.random.random((size, size))

        cp.cuda.Stream.null.synchronize()
        start = time.time()
        for _ in range(5):
            c = cp.dot(a, b)
            cp.cuda.Stream.null.synchronize()
        end = time.time()

        runtime = end - start
        ops_per_sec = (2 * size**3 * 5) / runtime
        return runtime, ops_per_sec

    def _benchmark_cupy_elementwise_ops(self) -> Tuple[float, float]:
        import cupy as cp

        size = 1_000_000
        a = cp.random.random(size)
        b = cp.random.random(size)

        cp.cuda.Stream.null.synchronize()
        start = time.time()
        for _ in range(10):
            c = a * b + a / b - cp.sqrt(a)
            cp.cuda.Stream.null.synchronize()
        end = time.time()

        runtime = end - start
        ops_per_sec = (4 * size * 10) / runtime
        return runtime, ops_per_sec

    def _run_opencl_benchmark(self, cpu_results: Dict[str, Any]) -> Dict[str, Any]:
        results = {"tests": [], "total_time": 0, "errors": [], "speedup": []}

        try:
            import pyopencl as cl
            import numpy as np

            platforms = cl.get_platforms()
            if not platforms:
                results["errors"].append("No OpenCL platforms found")
                return results

            ctx = cl.Context(dev_type=cl.device_type.GPU)
            if not ctx.devices:
                results["errors"].append("No GPU devices found for OpenCL")
                return results

            queue = cl.CommandQueue(ctx)

            size = 1000
            a_host = np.random.random((size, size)).astype(np.float32)
            b_host = np.random.random((size, size)).astype(np.float32)

            a_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_host
            )
            b_buf = cl.Buffer(
                ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_host
            )
            c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, a_host.nbytes)

            kernel = """
            __kernel void matmul(__global const float *A, __global const float *B, __global float *C, int N) {
                int row = get_global_id(0);
                int col = get_global_id(1);
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
            """

            prg = cl.Program(ctx, kernel).build()

            start = time.time()
            for _ in range(5):
                prg.matmul(
                    queue, (size, size), None, a_buf, b_buf, c_buf, np.int32(size)
                )
                queue.finish()
            end = time.time()

            runtime = end - start
            ops_per_sec = (2 * size**3 * 5) / runtime

            results["tests"].append(
                {
                    "name": "GPU - Matrix Multiplication (OpenCL)",
                    "time": runtime,
                    "ops_per_sec": ops_per_sec,
                }
            )

            cpu_test = next(
                (
                    t
                    for t in cpu_results["tests"]
                    if "matrix multiply" in t["name"].lower()
                ),
                None,
            )
            if cpu_test:
                results["speedup"].append(
                    {"test": "matrix_multiply", "speedup": cpu_test["time"] / runtime}
                )

            results["total_time"] = runtime

        except Exception as e:
            results["errors"].append(f"OpenCL benchmark error: {str(e)}")

        return results


def format_for_students(detector: HardwareDetector) -> str:
    output: List[str] = []

    output.append("=" * 60)
    output.append("HARDWARE ACCELERATION CHECK")
    output.append("=" * 60)
    output.append("")

    output.append("This tool checks your computer for special hardware that can")
    output.append("speed up tasks like graphics, AI, and scientific computing.")
    output.append("")

    cpu_info = detector.detect_cpu()
    output.append("-" * 60)
    output.append("CPU / PROCESSOR")
    output.append("-" * 60)
    if cpu_info:
        for key, value in cpu_info.items():
            if value:
                output.append(f"{key:20s}: {value}")
    else:
        output.append("Could not detect CPU information")
    output.append("")

    cpu_features = detector.detect_cpu_features()
    if cpu_features["vector_extensions"]:
        output.append("-" * 60)
        output.append("CPU SPEED-UP FEATURES")
        output.append("-" * 60)
        output.append("Your CPU has these special instructions for faster math:")
        for ext in cpu_features["vector_extensions"]:
            output.append(f"  {ext}")
        output.append("")

    gpu_info = detector.detect_gpu()
    output.append("-" * 60)
    output.append("GRAPHICS PROCESSING UNIT (GPU)")
    output.append("-" * 60)
    if gpu_info["present"]:
        output.append("YES - A graphics accelerator was detected!")
        output.append("")
        output.append("What this means:")
        output.append("  - Faster graphics and games")
        output.append("  - Can help with video processing")
        output.append("  - May speed up some AI tasks")
        output.append("")
        if gpu_info["devices"]:
            output.append("Devices found:")
            for dev in gpu_info["devices"][:5]:
                output.append(f"  {dev}")
        if gpu_info["modules"]:
            output.append("Driver modules:")
            for mod in gpu_info["modules"][:5]:
                output.append(f"  {mod}")
    else:
        output.append("NO graphics accelerator detected")
        output.append("Graphics will use your main CPU")
    output.append("")

    npu_info = detector.detect_npu()
    output.append("-" * 60)
    output.append("NEURAL PROCESSING UNIT (NPU) / AI CHIP")
    output.append("-" * 60)
    if npu_info["present"]:
        output.append("YES - An AI accelerator was detected!")
        output.append("")
        output.append("What this means:")
        output.append("  - Faster AI and machine learning tasks")
        output.append("  - Better performance for neural networks")
        output.append("  - Can speed up image recognition and language models")
        output.append("")
        if npu_info["details"]:
            output.append("Detection details:")
            for detail in npu_info["details"][:5]:
                output.append(f"  {detail}")
    else:
        output.append("NO dedicated AI accelerator detected")
        output.append("AI tasks will use your CPU and/or GPU")
    output.append("")

    opencl_info = detector.detect_opencl()
    output.append("-" * 60)
    output.append("OPENCL (GENERAL ACCELERATION)")
    output.append("-" * 60)
    if opencl_info["available"]:
        output.append("YES - OpenCL is available")
        output.append("")
        output.append("What this means:")
        output.append("  - Programs can use OpenCL to run on accelerators")
        output.append("  - Many scientific and graphics apps use OpenCL")
        output.append("  - Provides a way to access GPU/NPU power")
    else:
        output.append("NO OpenCL detected")
    output.append("")

    pci_devices = detector.detect_pci_devices()
    if pci_devices:
        output.append("-" * 60)
        output.append("PCIe ACCELERATION DEVICES")
        output.append("-" * 60)
        output.append("Found these add-on cards that can speed up work:")
        for dev in pci_devices[:5]:
            output.append(f"  {dev}")
        output.append("")

    usb_devices = detector.detect_usb_devices()
    if usb_devices:
        output.append("-" * 60)
        output.append("USB ACCELERATION DEVICES")
        output.append("-" * 60)
        output.append("Found these USB devices that can speed up work:")
        for dev in usb_devices[:5]:
            output.append(f"  {dev}")
        output.append("")

    storage_info = detector.detect_storage_accelerators()
    has_storage_accel = any(
        [
            storage_info["nvme"],
            storage_info["ssd"],
            storage_info["raid"],
            storage_info["encryption"],
        ]
    )
    if has_storage_accel:
        output.append("-" * 60)
        output.append("STORAGE ACCELERATION")
        output.append("-" * 60)
        if storage_info["nvme"]:
            output.append("NVMe storage detected - Very fast read/write speeds")
        if storage_info["ssd"]:
            output.append("SSD storage detected - Faster than traditional hard drives")
        if storage_info["raid"]:
            output.append(
                "RAID detected - Multiple drives working together for speed/redundancy"
            )
        if storage_info["encryption"]:
            output.append("Hardware encryption - Faster data encryption/decryption")
        if storage_info["details"]:
            for detail in storage_info["details"][:3]:
                output.append(f"  {detail}")
        output.append("")

    net_info = detector.detect_network_accelerators()
    has_net_accel = any([net_info["offload"], net_info["rdma"], net_info["dpdk"]])
    if has_net_accel:
        output.append("-" * 60)
        output.append("NETWORK ACCELERATION")
        output.append("-" * 60)
        if net_info["offload"]:
            output.append(
                "Network offload detected - Network card handles some processing"
            )
            output.append("  This reduces CPU load for network operations")
        if net_info["rdma"]:
            output.append(
                "RDMA detected - Direct memory access for very fast networking"
            )
            output.append("  Used in high-performance computing and data centers")
        if net_info["dpdk"]:
            output.append("DPDK support - Userspace networking for high speed")
            output.append("  Bypasses standard networking for maximum performance")
        if net_info["details"]:
            for detail in net_info["details"][:3]:
                output.append(f"  {detail}")
        output.append("")

    crypto_info = detector.detect_crypto_accelerators()
    if crypto_info["hardware"]:
        output.append("-" * 60)
        output.append("CRYPTOGRAPHIC ACCELERATION")
        output.append("-" * 60)
        output.append("Hardware crypto acceleration detected")
        output.append("What this means:")
        output.append("  - Faster encryption and decryption")
        output.append("  - Better performance for secure connections")
        output.append("  - Reduced CPU load for cryptographic operations")
        if crypto_info["modules"]:
            output.append("Modules:")
            for mod in crypto_info["modules"][:3]:
                output.append(f"  {mod}")
        if crypto_info["details"]:
            for detail in crypto_info["details"][:3]:
                output.append(f"  {detail}")
        output.append("")

    video_info = detector.detect_video_accelerators()
    has_video_accel = video_info["v4l2"] or len(video_info["codecs"]) > 0
    if has_video_accel:
        output.append("-" * 60)
        output.append("VIDEO ACCELERATION")
        output.append("-" * 60)
        if video_info["v4l2"]:
            output.append("Video devices detected - Hardware video capture/processing")
        if video_info["codecs"]:
            output.append("Hardware codec support for:")
            for codec in video_info["codecs"][:5]:
                output.append(f"  {codec}")
        output.append("")

    compress_info = detector.detect_compression_accelerators()
    if compress_info["hardware"]:
        output.append("-" * 60)
        output.append("COMPRESSION ACCELERATION")
        output.append("-" * 60)
        output.append("Hardware compression detected")
        output.append("What this means:")
        output.append("  - Faster data compression and decompression")
        output.append("  - Speeds up file transfers and storage operations")
        output.append("  - Reduces CPU load for compression tasks")
        if compress_info["details"]:
            for detail in compress_info["details"][:3]:
                output.append(f"  {detail}")
        output.append("")

    output.append("=" * 60)
    output.append("SUMMARY")
    output.append("=" * 60)

    accelerators: List[str] = []
    if gpu_info["present"]:
        accelerators.append("Graphics (GPU)")
    if npu_info["present"]:
        accelerators.append("AI/Neural (NPU)")
    if cpu_features["vector_extensions"]:
        accelerators.append("CPU Vector Extensions")
    if opencl_info["available"]:
        accelerators.append("OpenCL Support")
    if pci_devices or usb_devices:
        accelerators.append("Add-on Accelerator Cards")
    if storage_info["nvme"]:
        accelerators.append("NVMe Storage")
    if storage_info["raid"]:
        accelerators.append("RAID Storage")
    if storage_info["encryption"]:
        accelerators.append("Hardware Crypto/Encryption")
    if net_info["rdma"]:
        accelerators.append("RDMA Networking")
    if net_info["offload"]:
        accelerators.append("Network Offload")
    if net_info["dpdk"]:
        accelerators.append("DPDK Support")
    if crypto_info["hardware"]:
        accelerators.append("Cryptographic Acceleration")
    if video_info["v4l2"] or video_info["codecs"]:
        accelerators.append("Video Acceleration")
    if compress_info["hardware"]:
        accelerators.append("Compression Acceleration")

    if accelerators:
        output.append("Your system has these acceleration features:")
        for acc in accelerators:
            output.append(f"  {acc}")
        output.append("")
        output.append("This means some programs will run faster than on a basic")
        output.append("computer without these features.")
    else:
        output.append("Your system uses the basic CPU for all work.")
        output.append("You don't have special hardware acceleration, but your")
        output.append("CPU will still handle everything - it just might be slower")
        output.append("for graphics and AI tasks than systems with accelerators.")

    output.append("")
    output.append("=" * 60)

    return "\n".join(output)


def format_benchmarks_for_students(
    benchmark_runner: BenchmarkRunner, show_gpu: bool = False
) -> str:
    output: List[str] = []

    output.append("=" * 60)
    output.append("PERFORMANCE BENCHMARKS")
    output.append("=" * 60)
    output.append("")
    output.append("This section tests how fast your computer can do math and")
    output.append("data processing tasks. We measure how many operations per")
    output.append("second your computer can handle.")
    output.append("")

    numpy_cpu = benchmark_runner.run_numpy_cpu_benchmark()
    pandas_cpu = benchmark_runner.run_pandas_cpu_benchmark()

    if not numpy_cpu["available"] and not pandas_cpu["available"]:
        output.append("NumPy and Pandas are not installed. Cannot run benchmarks.")
        output.append("")
        output.append("To install: pip install numpy pandas")
        output.append("")
        output.append("=" * 60)
        return "\n".join(output)

    output.append("-" * 60)
    output.append("NUMPY (CPU) - MATH AND NUMBER CRUNCHING")
    output.append("-" * 60)
    output.append("NumPy is a library for fast number crunching in Python.")
    output.append("Think of it like a super-powered calculator.")
    output.append("")

    if numpy_cpu["available"]:
        output.append(
            f"Total time for all tests: {numpy_cpu['total_time']:.3f} seconds"
        )
        output.append("")
        output.append("Tests performed:")
        for test in numpy_cpu["tests"]:
            output.append(f"  {test['name']}")
            output.append(f"    Time: {test['time']:.3f} seconds")
            if test["ops_per_sec"] > 0:
                output.append(f"    Speed: {test['ops_per_sec']:.0f} operations/second")
            output.append("")

        if numpy_cpu["errors"]:
            output.append("Some tests had errors:")
            for error in numpy_cpu["errors"][:3]:
                output.append(f"  - {error}")
            output.append("")

    output.append("-" * 60)
    output.append("PANDAS (CPU) - DATA PROCESSING")
    output.append("-" * 60)
    output.append("Pandas is for working with tables of data, like spreadsheets.")
    output.append("It's used for data analysis, statistics, and data science.")
    output.append("")

    if pandas_cpu["available"]:
        output.append(
            f"Total time for all tests: {pandas_cpu['total_time']:.3f} seconds"
        )
        output.append("")
        output.append("Tests performed:")
        for test in pandas_cpu["tests"]:
            output.append(f"  {test['name']}")
            output.append(f"    Time: {test['time']:.3f} seconds")
            if test["ops_per_sec"] > 0:
                output.append(f"    Speed: {test['ops_per_sec']:.0f} rows/second")
            output.append("")

        if pandas_cpu["errors"]:
            output.append("Some tests had errors:")
            for error in pandas_cpu["errors"][:3]:
                output.append(f"  - {error}")
            output.append("")

    output.append("-" * 60)
    output.append("WHAT THESE NUMBERS MEAN")
    output.append("-" * 60)
    output.append("Think of these tests like sports drills:")
    output.append("")
    output.append("  Matrix Multiplication: Like calculating many math problems")
    output.append("                       at once. Important for AI and physics.")
    output.append("")
    output.append("  Element-wise Operations: Simple math on many numbers.")
    output.append("                         Like adding two big lists together.")
    output.append("")
    output.append("  Linear Algebra: Complex math used in 3D graphics and")
    output.append("                  solving systems of equations.")
    output.append("")
    output.append("  FFT: Transforms data to find patterns. Used in music")
    output.append("       processing, image compression, and science.")
    output.append("")
    output.append("  DataFrame Operations: Working with tables of data.")
    output.append("                         Like Excel but much faster.")
    output.append("")

    numpy_gpu = None
    if show_gpu:
        numpy_gpu = benchmark_runner.run_numpy_gpu_benchmark()

        output.append("=" * 60)
        output.append("GPU ACCELERATION TEST")
        output.append("=" * 60)
        output.append("")

        if numpy_gpu["available"] and numpy_gpu["tests"]:
            output.append(f"Method: {numpy_gpu['method']}")
            output.append("")
            output.append("GPU Tests performed:")
            for test in numpy_gpu["tests"]:
                output.append(f"  {test['name']}")
                output.append(f"    Time: {test['time']:.3f} seconds")
                if test["ops_per_sec"] > 0:
                    output.append(
                        f"    Speed: {test['ops_per_sec']:.0f} operations/second"
                    )
            output.append("")

            if numpy_gpu["speedup"]:
                output.append("Speedup Comparison (GPU vs CPU):")
                output.append("")
                for speedup in numpy_gpu["speedup"]:
                    if speedup["speedup"] > 0:
                        times_faster = speedup["speedup"]
                        output.append(f"  {speedup['test'].replace('_', ' ').title()}:")
                        output.append(f"    GPU is {times_faster:.1f}x faster than CPU")
                        if times_faster > 10:
                            output.append("    WOW! That's a huge speedup!")
                        elif times_faster > 2:
                            output.append("    That's noticeably faster!")
                        else:
                            output.append("    The GPU helps a little.")
                        output.append("")
        else:
            output.append("GPU acceleration test could not be completed.")
            output.append("")
            if numpy_gpu["errors"]:
                output.append("Reasons:")
                for error in numpy_gpu["errors"][:3]:
                    output.append(f"  - {error}")
                output.append("")
            output.append("This might mean:")
            output.append("  - No compatible GPU was found")
            output.append("  - GPU acceleration libraries are not installed")
            output.append("  - Your GPU doesn't support the required features")
            output.append("  - You need to install: pip install cupy")
            output.append("")

    output.append("=" * 60)
    output.append("SUMMARY")
    output.append("=" * 60)
    output.append("")

    if numpy_cpu["available"] and pandas_cpu["available"]:
        total_cpu_time = numpy_cpu["total_time"] + pandas_cpu["total_time"]
        output.append(
            f"CPU Processing Time: {total_cpu_time:.3f} seconds for all tests"
        )
        output.append("")

        if show_gpu and numpy_gpu and numpy_gpu["available"] and numpy_gpu["tests"]:
            gpu_time = numpy_gpu["total_time"]
            output.append(
                f"GPU Processing Time: {gpu_time:.3f} seconds for available tests"
            )
            output.append("")

            if numpy_gpu["speedup"]:
                avg_speedup = sum(
                    s["speedup"] for s in numpy_gpu["speedup"] if s["speedup"] > 0
                ) / len([s for s in numpy_gpu["speedup"] if s["speedup"] > 0])
                output.append(
                    f"Average GPU Speedup: {avg_speedup:.1f}x faster than CPU"
                )
                output.append("")

                if avg_speedup > 5:
                    output.append(
                        "Your GPU is doing an excellent job accelerating computations!"
                    )
                    output.append("Tasks that used to take minutes now take seconds.")
                elif avg_speedup > 1.5:
                    output.append("Your GPU is helping speed up computations.")
                    output.append("You'll notice the difference on larger tasks.")
                else:
                    output.append("Your GPU is helping, but the speedup is modest.")
                    output.append("For this size of data, CPU vs GPU is similar.")
                    output.append("With larger datasets, GPU would shine more.")
            else:
                output.append("GPU speedup could not be measured.")

        output.append("")
        output.append("How to think about these results:")
        output.append("")
        output.append("  - Faster times mean your computer can process more data")
        output.append("    in the same amount of time")
        output.append("")
        output.append("  - GPU acceleration helps most with large datasets")
        output.append("    (thousands or millions of items)")
        output.append("")
        output.append("  - For small datasets, your CPU might be just as fast")
        output.append("    because of the time needed to copy data to the GPU")
        output.append("")
        output.append("  - The best setup depends on what you're doing:")
        output.append("    * AI/Machine Learning: GPU makes a huge difference")
        output.append("    * Data Analysis: CPU often fine for small datasets")
        output.append("    * Scientific Computing: GPU can be much faster")
        output.append("    * General Programming: CPU is usually sufficient")
        output.append("")
    else:
        output.append("Could not complete benchmarks. Please install:")
        output.append("  pip install numpy pandas")
        output.append("")

    output.append("=" * 60)

    return "\n".join(output)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Hardware Acceleration Detector and Benchmark Tool"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )
    parser.add_argument(
        "--gpu-benchmark", action="store_true", help="Run GPU acceleration benchmarks"
    )
    args = parser.parse_args()

    detector = HardwareDetector()
    print(format_for_students(detector))

    if args.benchmark or args.gpu_benchmark:
        print("")
        benchmark_runner = BenchmarkRunner()
        print(
            format_benchmarks_for_students(
                benchmark_runner, show_gpu=args.gpu_benchmark
            )
        )


if __name__ == "__main__":
    main()
