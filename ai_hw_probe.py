#!/usr/bin/env python3

import os
import re
import shutil
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
            threads = 0
            cores = 0
            sockets = 0
            for line in cpuinfo.split("\n"):
                if line.startswith("processor"):
                    threads += 1
                elif line.startswith("cpu cores"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        cores = int(parts[1].strip())
                elif line.startswith("physical id"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        socket_id = int(parts[1].strip())
                        if socket_id + 1 > sockets:
                            sockets = socket_id + 1
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

            if threads > 0:
                info["threads"] = str(threads)
            if cores > 0:
                if sockets > 1:
                    cores *= sockets
                info["cores"] = str(cores)

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
            "make_model": [],
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

        if results["present"]:
            results["make_model"] = self._extract_make_model("gpu", results)

        return results

    def detect_npu(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "present": False,
            "devices": [],
            "modules": [],
            "libraries": [],
            "details": [],
            "make_model": [],
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

        if results["present"]:
            results["make_model"] = self._extract_make_model("npu", results)

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

    def _extract_make_model(
        self, device_type: str, results: Dict[str, Any]
    ) -> List[str]:
        make_models: List[str] = []

        if device_type == "gpu":
            pci_devices = self.detect_pci_devices()
            for device in pci_devices:
                if re.search(
                    r"NVIDIA|AMD|Intel.*Graphics|Radeon|GeForce|Quadro|Tesla|FirePro",
                    device,
                    re.IGNORECASE,
                ):
                    make_models.append(device)

            drm_devices = self.detect_drm_devices()
            for dev in drm_devices:
                if "uevent" in dev:
                    uevent_content = dev["uevent"]
                    if re.search(r"NVIDIA|AMD|INTEL", uevent_content, re.IGNORECASE):
                        if "DRIVER" in uevent_content:
                            driver_match = re.search(r"DRIVER=([^\s]+)", uevent_content)
                            if driver_match:
                                make_models.append(f"Driver: {driver_match.group(1)}")
                        if "PCI_ID" in uevent_content:
                            pci_match = re.search(r"PCI_ID=([^\s]+)", uevent_content)
                            if pci_match:
                                make_models.append(f"PCI ID: {pci_match.group(1)}")

            if results["modules"]:
                for module in results["modules"]:
                    module_info = {
                        "etnaviv": "Vivante GPU (etnaviv driver)",
                        "galcore": "Vivante GC GPU (galcore driver)",
                        "vsi": "Vivante/OpenVX GPU",
                        "imx_gpu": "NXP i.MX GPU",
                    }
                    if module in module_info:
                        make_models.append(module_info[module])
                    else:
                        make_models.append(f"{module} GPU module")

        elif device_type == "npu":
            if results["modules"]:
                module_info = {
                    "ethos": "Arm Ethos NPU",
                    "vsi": "VeriSilicon/OpenVX NPU",
                    "tpu": "Tensor Processing Unit (TPU)",
                    "nvdla": "NVIDIA Deep Learning Accelerator (NVDLA)",
                    "openvit": "OpenVIT NPU",
                }
                for module in results["modules"]:
                    if module in module_info:
                        make_models.append(module_info[module])
                    else:
                        make_models.append(f"{module} NPU module")

            usb_devices = self.detect_usb_devices()
            for device in usb_devices:
                if re.search(r"Coral|TPU|Google|GlobalUnichip", device, re.IGNORECASE):
                    make_models.append(device)

        return make_models

    def detect_memory(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "total_gb": 0,
            "available_gb": 0,
            "used_gb": 0,
            "swap_gb": 0,
            "details": [],
        }

        meminfo = self._read_file("/proc/meminfo")
        if meminfo:
            mem_total = 0
            mem_available = 0
            swap_total = 0

            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1])
                elif line.startswith("SwapTotal:"):
                    swap_total = int(line.split()[1])

            if mem_total > 0:
                results["total_gb"] = mem_total / (1024 * 1024)
            if mem_available > 0:
                results["available_gb"] = mem_available / (1024 * 1024)
                results["used_gb"] = results["total_gb"] - results["available_gb"]
            if swap_total > 0:
                results["swap_gb"] = swap_total / (1024 * 1024)

            if results["total_gb"] > 0:
                results["details"].append(f"Total RAM: {results['total_gb']:.2f} GB")
                results["details"].append(
                    f"Available: {results['available_gb']:.2f} GB"
                )
                results["details"].append(f"Used: {results['used_gb']:.2f} GB")
                if results["swap_gb"] > 0:
                    results["details"].append(f"Swap: {results['swap_gb']:.2f} GB")

        return results

    def detect_disk(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "total_gb": 0,
            "available_gb": 0,
            "used_gb": 0,
            "swap_partitions": [],
            "details": [],
        }

        # Get disk usage for the partition where this script is running
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        try:
            usage = shutil.disk_usage(script_dir)
            results["total_gb"] = usage.total / (1024**3)
            results["available_gb"] = usage.free / (1024**3)
            results["used_gb"] = usage.used / (1024**3)
        except Exception:
            pass

        # Check for swap partitions
        try:
            with open("/proc/swaps", "r") as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.split()
                    if len(parts) >= 5 and parts[0] != "Filename":
                        size_kb = int(parts[2])
                        if size_kb > 0:
                            results["swap_partitions"].append(
                                {
                                    "device": parts[0],
                                    "type": parts[1],
                                    "size_gb": size_kb / (1024 * 1024),
                                }
                            )
        except Exception:
            pass

        if results["total_gb"] > 0:
            results["details"].append(f"Total Disk Space: {results['total_gb']:.2f} GB")
            results["details"].append(
                f"Available Space: {results['available_gb']:.2f} GB"
            )
            results["details"].append(f"Used Space: {results['used_gb']:.2f} GB")

        if results["swap_partitions"]:
            total_swap = sum(p["size_gb"] for p in results["swap_partitions"])
            results["details"].append(
                f"Swap Partitions: {len(results['swap_partitions'])} ({total_swap:.2f} GB total)"
            )

        return results

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

    def detect_gpu_device_details(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "render_devices": [],
            "drm_devices": [],
            "dri_cards": [],
            "device_capabilities": {},
            "opencl_platforms": [],
            "vulkan_available": False,
            "opengl_version": None,
            "rendering_backend": None,
            "vendor_info": {},
            "is_software_rendering": False,
        }

        vendor_id_map = {
            "0x1af4": {
                "name": "Virtio",
                "driver": "virtio-gpu",
                "description": "Virtio GPU (VM paravirtualized)",
            },
            "0x15ad": {
                "name": "VMware",
                "driver": "vmwgfx",
                "description": "VMware SVGA GPU (VM virtualized)",
            },
            "0x8086": {
                "name": "Intel",
                "driver": "i915 / Xe",
                "description": "Intel Graphics (integrated/dedicated)",
            },
            "0x1002": {
                "name": "AMD",
                "driver": "amdgpu",
                "description": "AMD Radeon Graphics",
            },
            "0x10de": {
                "name": "NVIDIA",
                "driver": "nvidia/nouveau",
                "description": "NVIDIA Graphics (CUDA uses /dev/nvidia*, Vulkan via nvk/mesa)",
            },
        }

        dri_path = "/dev/dri"
        if self._file_exists(dri_path):
            try:
                devices = os.listdir(dri_path)
                for dev in devices:
                    dev_path = os.path.join(dri_path, dev)
                    if dev.startswith("renderD"):
                        results["render_devices"].append(dev_path)
                        cap_info = self._get_device_capabilities(dev_path)

                        pci_id = cap_info.get("pci_id", "")
                        if pci_id:
                            vendor_hex = pci_id.split(":")[0].strip()
                            if vendor_hex in vendor_id_map:
                                vendor_info = vendor_id_map[vendor_hex]
                                results["vendor_info"][dev] = vendor_info
                                cap_info["vendor_name"] = vendor_info["name"]
                                cap_info["vendor_description"] = vendor_info[
                                    "description"
                                ]

                        results["device_capabilities"][dev] = cap_info
                    elif dev.startswith("card"):
                        results["dri_cards"].append(dev_path)

                card_path = "/sys/class/drm"
                for card in Path(card_path).glob("card*"):
                    if card.is_dir():
                        results["drm_devices"].append(str(card))

            except Exception:
                pass

        if self._command_available("vulkaninfo"):
            code, output = self._run_command(["vulkaninfo", "--summary"])
            if code == 0:
                results["vulkan_available"] = True
                version_match = re.search(r"Vulkan Version:\s*([0-9.]+)", output)
                if version_match:
                    results["opengl_version"] = version_match.group(1)

        if self._command_available("glxinfo"):
            code, output = self._run_command(["glxinfo"])
            if code == 0:
                version_match = re.search(r"OpenGL version string:\s*(.+)", output)
                if version_match:
                    results["opengl_version"] = version_match.group(1)

                renderer_match = re.search(r"OpenGL renderer string:\s*(.+)", output)
                if renderer_match:
                    renderer = renderer_match.group(1)
                    results["rendering_backend"] = renderer

                    if "llvmpipe" in renderer.lower():
                        results["is_software_rendering"] = True

                vendor_match = re.search(r"OpenGL vendor string:\s*(.+)", output)
                if vendor_match:
                    results["vendor_info"]["glx_vendor"] = vendor_match.group(1)

        if self._command_available("glxinfo"):
            code, output = self._run_command(["glxinfo", "-B"])
            if code == 0 and "llvmpipe" in output.lower():
                results["is_software_rendering"] = True

        return results

    def _get_device_capabilities(self, device_path: str) -> Dict[str, Any]:
        caps: Dict[str, Any] = {
            "vendor": None,
            "device": None,
            "driver": None,
            "compute_units": None,
            "max_freq": None,
            "memory_size": None,
        }

        try:
            stat_info = os.stat(device_path)
            major = os.major(stat_info.st_rdev)
            minor = os.minor(stat_info.st_rdev)

            uevent_path = f"/sys/dev/char/{major}:{minor}/uevent"
            uevent = self._read_file(uevent_path)
            if uevent:
                for line in uevent.split("\n"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        if key in ["DRIVER", "PCI_ID", "PCI_SUBSYS_ID"]:
                            caps[key.lower()] = value

        except Exception:
            pass

        return caps

    def detect_virtio_components(self, in_vm: bool) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "in_vm": in_vm,
            "components": {},
            "details": [],
        }

        if not in_vm:
            return results

        virtio_components = {
            "virtio-blk": "Virtio Block Storage (fast disk I/O)",
            "virtio-net": "Virtio Network (fast network I/O)",
            "virtio-gpu": "Virtio GPU (virtualized graphics)",
            "virtio-balloon": "Virtio Balloon (memory management)",
            "virtio-console": "Virtio Console (VM communication)",
            "virtio-rng": "Virtio RNG (random number generation)",
            "virtio-serial": "Virtio Serial (VM communication)",
            "virtio-scsi": "Virtio SCSI (storage)",
            "virtio-9p": "Virtio 9P (file sharing)",
            "virtio-fs": "Virtio FS (fast file sharing)",
        }

        if self._command_available("lspci"):
            code, output = self._run_command(["lspci", "-nn", "-v"])
            if code == 0:
                for virtio, desc in virtio_components.items():
                    if re.search(virtio, output, re.IGNORECASE) is not None:
                        results["components"][virtio] = desc
                        results["details"].append(f"{virtio}: {desc}")

        virtio_path = "/sys/devices/virtual"
        if self._file_exists(virtio_path):
            try:
                for item in Path(virtio_path).iterdir():
                    if "virtio" in item.name.lower():
                        comp_name = item.name
                        if comp_name in virtio_components:
                            results["components"][comp_name] = virtio_components[
                                comp_name
                            ]
                        results["details"].append(f"Found: {comp_name}")
            except Exception:
                pass

        if self._command_available("lsmod"):
            code, output = self._run_command(["lsmod"])
            if code == 0:
                for virtio, desc in virtio_components.items():
                    module = virtio.replace("-", "_")
                    if re.search(module, output, re.IGNORECASE) is not None:
                        if virtio not in results["components"]:
                            results["components"][virtio] = desc
                            results["details"].append(f"{virtio}: {desc}")

        if self.dmesg:
            if re.search(r"virtio", self.dmesg, re.IGNORECASE) is not None:
                virtio_matches = re.findall(
                    r"virtio[_-][a-z0-9]+", self.dmesg, re.IGNORECASE
                )
                for match in virtio_matches:
                    match_clean = match.replace("_", "-")
                    if match_clean in virtio_components:
                        if match_clean not in results["components"]:
                            results["components"][match_clean] = virtio_components[
                                match_clean
                            ]
                            results["details"].append(
                                f"{match_clean}: detected in dmesg"
                            )

        return results

    def detect_vfio(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "available": False,
            "modules_loaded": [],
            "vfio_devices": [],
            "iommu_groups": [],
            "iommu_devices": {},
            "platform_devices": [],
            "mdev_devices": [],
            "devices_bound_to_vfio": [],
            "details": [],
        }

        vfio_modules = [
            "vfio",
            "vfio_pci",
            "vfio_iommu_type1",
            "vfio_iommu_type2",
            "vfio_virqfd",
            "vfio_mdev",
            "vfio_fsl_mc",
        ]

        if self._command_available("lsmod"):
            code, output = self._run_command(["lsmod"])
            if code == 0:
                for module in vfio_modules:
                    if re.search(module, output, re.IGNORECASE) is not None:
                        results["available"] = True
                        results["modules_loaded"].append(module)

        vfio_path = "/dev/vfio"
        if self._file_exists(vfio_path):
            try:
                devices = os.listdir(vfio_path)
                for dev in devices:
                    if re.match(r"^[0-9]+$", dev):
                        results["vfio_devices"].append(f"/dev/vfio/{dev}")
                if results["vfio_devices"]:
                    results["available"] = True
                    results["details"].append(
                        f"Found {len(results['vfio_devices'])} VFIO device(s)"
                    )
            except Exception:
                pass

        iommu_group_path = "/sys/kernel/iommu_groups"
        if self._file_exists(iommu_group_path):
            try:
                groups = list(Path(iommu_group_path).iterdir())
                for group in groups:
                    if group.is_dir() and group.name.isdigit():
                        group_num = int(group.name)
                        group_devices = []
                        try:
                            devices = (group / "devices").iterdir()
                            for dev in devices:
                                if dev.is_dir():
                                    group_devices.append(dev.name)
                                    results["iommu_devices"][group_num] = group_devices
                        except Exception:
                            pass
                        if group_devices:
                            results["iommu_groups"].append(group_num)

                if results["iommu_groups"]:
                    results["details"].append(
                        f"Found {len(results['iommu_groups'])} IOMMU group(s)"
                    )
            except Exception:
                pass

        pci_devices_path = "/sys/bus/pci/devices"
        if self._file_exists(pci_devices_path):
            try:
                for device in Path(pci_devices_path).iterdir():
                    if device.is_dir():
                        driver_path = device / "driver"
                        if driver_path.exists():
                            try:
                                driver_name = driver_path.resolve().name
                                if "vfio" in driver_name.lower():
                                    results["available"] = True
                                    results["devices_bound_to_vfio"].append(device.name)
                                    results["details"].append(
                                        f"Device bound to VFIO: {device.name}"
                                    )
                            except Exception:
                                pass
            except Exception:
                pass

        vfio_platform_path = "/sys/bus/platform/devices"
        if self._file_exists(vfio_platform_path):
            try:
                for device in Path(vfio_platform_path).iterdir():
                    if device.is_dir() and "vfio" in device.name.lower():
                        results["available"] = True
                        results["platform_devices"].append(device.name)
                        results["details"].append(
                            f"VFIO platform device: {device.name}"
                        )
            except Exception:
                pass

        mdev_path = "/sys/bus/mdev/devices"
        if self._file_exists(mdev_path):
            try:
                for device in Path(mdev_path).iterdir():
                    if device.is_dir():
                        results["available"] = True
                        results["mdev_devices"].append(device.name)
                        results["details"].append(
                            f"VFIO mediated device: {device.name}"
                        )
            except Exception:
                pass

        if self.dmesg:
            if (
                re.search(
                    r"vfio-pci.*bound|vfio_iommu|iommu.*vfio", self.dmesg, re.IGNORECASE
                )
                is not None
            ):
                results["available"] = True
                results["details"].append("VFIO activity detected in dmesg")

            vfio_bind_matches = re.findall(
                r"vfio-pci.*bound to ([0-9a-fA-F:\.]+)", self.dmesg
            )
            for pci_addr in vfio_bind_matches:
                if pci_addr not in results["devices_bound_to_vfio"]:
                    results["devices_bound_to_vfio"].append(pci_addr)
                    results["details"].append(f"Device bound to VFIO: {pci_addr}")

        if self._command_available("lspci"):
            code, output = self._run_command(["lspci", "-nn", "-k"])
            if code == 0:
                for line in output.split("\n"):
                    if "Kernel driver in use" in line and "vfio" in line.lower():
                        pci_addr_match = re.search(r"^([0-9a-fA-F:\.]+):", line)
                        if pci_addr_match:
                            pci_addr = pci_addr_match.group(1)
                            if pci_addr not in results["devices_bound_to_vfio"]:
                                results["devices_bound_to_vfio"].append(pci_addr)
                                results["details"].append(
                                    f"Device using VFIO: {pci_addr}"
                                )

        return results

    def detect_required_software(
        self, gpu_info: Dict[str, Any], npu_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "gpu_software": [],
            "npu_software": [],
            "drivers": [],
            "python_packages": [],
        }

        if gpu_info["present"]:
            has_nvidia = False
            has_amd = False
            has_intel = False
            has_vivante = False
            has_opencl = False

            for mm in gpu_info.get("make_model", []):
                if re.search(r"NVIDIA|GeForce|Quadro|Tesla", mm, re.IGNORECASE):
                    has_nvidia = True
                elif re.search(r"AMD|Radeon|FirePro", mm, re.IGNORECASE):
                    has_amd = True
                elif re.search(r"Intel.*Graphics|Arc", mm, re.IGNORECASE):
                    has_intel = True
                elif re.search(r"Vivante|galcore|etnaviv", mm, re.IGNORECASE):
                    has_vivante = True

            if gpu_info["libraries"]:
                has_opencl = True

            if has_nvidia:
                results["drivers"].extend(
                    [
                        "NVIDIA proprietary driver (nvidia-driver)",
                        "NVIDIA CUDA Toolkit (cuda-toolkit)",
                    ]
                )
                results["gpu_software"].extend(
                    [
                        "PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                        "TensorFlow GPU: pip install tensorflow[and-cuda]",
                        "CuPy (GPU NumPy): pip install cupy-cuda11x or cupy-cuda12x",
                    ]
                )

            if has_amd:
                results["drivers"].extend(
                    [
                        "AMDGPU driver (for modern AMD GPUs)",
                        "ROCm software stack (for compute support)",
                    ]
                )
                results["gpu_software"].extend(
                    [
                        "PyTorch with ROCm: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7",
                        "TensorFlow ROCm: pip install tensorflow-rocm",
                    ]
                )

            if has_intel:
                results["drivers"].extend(
                    [
                        "Intel GPU drivers (i915 for integrated, iris for Arc)",
                    ]
                )
                results["gpu_software"].extend(
                    [
                        "Intel Extension for PyTorch: pip install intel_extension_for_pytorch",
                        "oneAPI Base Toolkit (from Intel)",
                    ]
                )

            if has_vivante:
                results["drivers"].extend(
                    [
                        "Galcore driver for Vivante GPUs",
                        "Etnaviv open-source driver (for older Vivante GPUs)",
                    ]
                )

            if has_opencl or has_vivante:
                results["gpu_software"].append("PyOpenCL: pip install pyopencl")

        if npu_info["present"]:
            has_ethos = False
            has_vsi = False
            has_coral = False
            has_nvdla = False

            for mm in npu_info.get("make_model", []):
                if re.search(r"Ethos|Arm.*NPU", mm, re.IGNORECASE):
                    has_ethos = True
                elif re.search(r"VSI|OpenVX", mm, re.IGNORECASE):
                    has_vsi = True
                elif re.search(r"Coral|TPU|Google.*Edge", mm, re.IGNORECASE):
                    has_coral = True
                elif re.search(r"NVDLA|NVIDIA.*DLA", mm, re.IGNORECASE):
                    has_nvdla = True

            for module in npu_info["modules"]:
                if "ethos" in module.lower():
                    has_ethos = True
                elif "vsi" in module.lower():
                    has_vsi = True
                elif "nvdla" in module.lower():
                    has_nvdla = True

            if has_ethos:
                results["drivers"].append("Arm Ethos-U NPU kernel driver")
                results["npu_software"].extend(
                    [
                        "TensorFlow Lite with NPU delegate",
                        "Arm Ethos-U NPU software stack",
                        "Vela compiler for Ethos-U",
                    ]
                )

            if has_vsi:
                results["drivers"].append("VeriSilicon/VSI NPU driver")
                results["npu_software"].extend(
                    [
                        "VeriSilicon OpenVX runtime",
                        "NPU acceleration libraries from your device vendor",
                    ]
                )

            if has_coral:
                results["drivers"].append("libedgetpu (Google Edge TPU driver)")
                results["npu_software"].extend(
                    [
                        "PyCoral: pip install pycoral",
                        "TensorFlow Lite for Edge TPU",
                    ]
                )

            if has_nvdla:
                results["drivers"].append("NVIDIA NVDLA kernel driver")
                results["npu_software"].extend(
                    [
                        "NVIDIA NVDLA runtime libraries",
                        "TensorRT for NVDLA (if supported)",
                    ]
                )

        if gpu_info["present"] or npu_info["present"]:
            results["python_packages"].extend(
                [
                    "OpenCL support: pip install pyopencl",
                    "CUDA/ROCm packages (choose based on your GPU):",
                    "  - For NVIDIA: cupy, torch[cuda], tensorflow[and-cuda]",
                    "  - For AMD: torch[rocm], tensorflow-rocm",
                    "  - For Intel: intel_extension_for_pytorch",
                ]
            )

        return results

    def get_practical_usage_examples(
        self,
        gpu_info: Dict[str, Any],
        npu_info: Dict[str, Any],
        gpu_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "examples": [],
            "test_commands": [],
            "verification": [],
        }

        if gpu_info["present"] and gpu_details["render_devices"]:
            render_dev = gpu_details["render_devices"][0]
            driver = (
                gpu_details["device_capabilities"]
                .get(render_dev, {})
                .get("driver", "unknown")
            )

            results["examples"].append("Python GPU Access Examples:")
            results["examples"].append("")

            if gpu_details["vulkan_available"]:
                results["examples"].append("1. Vulkan-based GPU compute:")
                results["examples"].append("   from vulkan import *")
                results["examples"].append("   instance = vkCreateInstance(...)")
                results["examples"].append(
                    "   # Can use for compute shaders and graphics"
                )
                results["examples"].append("")

            results["examples"].append(
                "2. OpenCL (works with most GPUs including virtio-gpu):"
            )
            results["examples"].append("   import pyopencl as cl")
            results["examples"].append("   platforms = cl.get_platforms()")
            results["examples"].append("   devices = platforms[0].get_devices()")
            results["examples"].append("   ctx = cl.Context([devices[0]])")
            results["examples"].append("   queue = cl.CommandQueue(ctx)")
            results["examples"].append("   # Use for GPU-accelerated computations")
            results["examples"].append("")

            results["examples"].append(
                "3. Direct device access (limited, mainly for rendering):"
            )
            results["examples"].append("   import pyglet")
            results["examples"].append("   window = pyglet.Window()")
            results["examples"].append("   # OpenGL rendering to virtual GPU")
            results["examples"].append("")

            results["test_commands"].append("Test GPU accessibility:")
            results["test_commands"].append(f"  ls -la {render_dev}")
            results["test_commands"].append("  ls -la /dev/dri/")
            results["test_commands"].append(
                "  pip install pyopencl && python -c \"import pyopencl; print('OpenCL:', len(cl.get_platforms()), 'platforms')\""
            )

            results["verification"].append("AI workloads with OpenCL:")
            results["verification"].append("  import pyopencl as cl")
            results["verification"].append("  import numpy as np")
            results["verification"].append(
                "  a = np.random.rand(1000000).astype(np.float32)"
            )
            results["verification"].append(
                "  ctx = cl.Context(cl.get_platforms()[0].get_devices())"
            )
            results["verification"].append("  queue = cl.CommandQueue(ctx)")
            results["verification"].append("  mf = cl.mem_flags")
            results["verification"].append(
                "  a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)"
            )
            results["verification"].append("  # Kernel execution on GPU")

        if npu_info["present"]:
            results["examples"].append("NPU Usage Examples:")
            results["examples"].append("")

            for mm in npu_info.get("make_model", []):
                if "Coral" in mm or "Edge TPU" in mm:
                    results["examples"].append("Google Edge TPU:")
                    results["examples"].append(
                        "  from pycoral.utils.edgetpu import make_interpreter"
                    )
                    results["examples"].append(
                        "  interpreter = make_interpreter('model.tflite')"
                    )
                    results["examples"].append("  interpreter.allocate_tensors()")
                    results["examples"].append("  # Run inference on Edge TPU")
                    results["examples"].append("")

                elif "Ethos" in mm:
                    results["examples"].append("Arm Ethos NPU:")
                    results["examples"].append(
                        "  import tflite_runtime.interpreter as tflite"
                    )
                    results["examples"].append(
                        "  delegate = tflite.load_delegate('libethosu_delegate.so')"
                    )
                    results["examples"].append(
                        "  interpreter = tflite.Interpreter('model.tflite', [delegate])"
                    )
                    results["examples"].append("  # Run inference on Ethos NPU")
                    results["examples"].append("")

        results["examples"].append(
            "General AI Libraries (work with GPU/NPU when available):"
        )
        results["examples"].append("  import torch")
        results["examples"].append(
            "  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
        )
        results["examples"].append("  model = model.to(device)")
        results["examples"].append("")
        results["examples"].append("  import tensorflow as tf")
        results["examples"].append(
            "  print('GPU devices:', tf.config.list_physical_devices('GPU'))"
        )
        results["examples"].append("")

        return results

    def detect_virtualization(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "in_vm": False,
            "in_container": False,
            "vm_type": None,
            "container_type": None,
            "vt_x": False,
            "vt_d": False,
            "iommu": False,
            "iommu_passthrough": False,
            "sr_iov": False,
            "amd_v": False,
            "amd_vi": False,
            "sev": False,
            "sev_es": False,
            "details": [],
        }

        dmi_product = self._read_file("/sys/class/dmi/id/product_name")
        dmi_sys_vendor = self._read_file("/sys/class/dmi/id/sys_vendor")
        cpuinfo = self._read_file("/proc/cpuinfo")
        cmdline = self._read_file("/proc/cmdline")
        cgroup = self._read_file("/proc/1/cgroup")

        cpu_flags = ""
        if cpuinfo:
            flags_match = re.search(r"flags\s*:\s*(.+)", cpuinfo, re.IGNORECASE)
            if flags_match:
                cpu_flags = flags_match.group(1)

        vm_indicators = [
            "vmware",
            "virtualbox",
            "kvm",
            "qemu",
            "xen",
            "hyper-v",
            "bochs",
            "parallels",
        ]

        if dmi_product:
            for vm in vm_indicators:
                if re.search(vm, dmi_product, re.IGNORECASE):
                    results["in_vm"] = True
                    results["vm_type"] = vm
                    results["details"].append(
                        f"VM detected in DMI product: {dmi_product}"
                    )
                    break

        if not results["in_vm"] and dmi_sys_vendor:
            for vm in vm_indicators:
                if re.search(vm, dmi_sys_vendor, re.IGNORECASE):
                    results["in_vm"] = True
                    results["vm_type"] = vm
                    results["details"].append(
                        f"VM detected in DMI vendor: {dmi_sys_vendor}"
                    )
                    break

        if self.dmesg and not results["in_vm"]:
            if re.search(r"hypervisor", self.dmesg, re.IGNORECASE) is not None:
                results["in_vm"] = True
                results["details"].append("Hypervisor detected in dmesg")

        if self._command_available("systemd-detect-virt"):
            code, output = self._run_command(["systemd-detect-virt"])
            if code == 0 and output.strip() and output.strip() != "none":
                results["in_vm"] = True
                if not results["vm_type"]:
                    results["vm_type"] = output.strip()
                results["details"].append(f"systemd-detect-virt: {output.strip()}")

        if cpu_flags:
            if "vmx" in cpu_flags:
                results["vt_x"] = True
                results["details"].append("Intel VT-x (CPU virtualization) supported")

            if "svm" in cpu_flags:
                results["amd_v"] = True
                results["details"].append("AMD-V (CPU virtualization) supported")

            if "dmar" in cpu_flags:
                results["vt_d"] = True
                results["details"].append("Intel VT-d (IOMMU) supported")

        if self.dmesg:
            if (
                re.search(r"DMAR|IOMMU enabled|AMD-Vi", self.dmesg, re.IGNORECASE)
                is not None
            ):
                results["iommu"] = True
                results["details"].append("IOMMU detected in dmesg")

            if (
                re.search(r"passthrough|DMA remapping", self.dmesg, re.IGNORECASE)
                is not None
            ):
                results["iommu_passthrough"] = True
                results["details"].append("IOMMU passthrough detected")

            if (
                re.search(
                    r"SR-IOV|Single Root I/O Virtualization", self.dmesg, re.IGNORECASE
                )
                is not None
            ):
                results["sr_iov"] = True
                results["details"].append("SR-IOV capability detected")

            if re.search(r"AMD-Vi", self.dmesg, re.IGNORECASE) is not None:
                results["amd_vi"] = True
                results["details"].append("AMD-Vi (IOMMU) detected")

            if re.search(r"SEV: Enabled", self.dmesg, re.IGNORECASE) is not None:
                results["sev"] = True
                results["details"].append(
                    "AMD SEV (Secure Encrypted Virtualization) detected"
                )

            if re.search(r"SEV-ES: Enabled", self.dmesg, re.IGNORECASE) is not None:
                results["sev_es"] = True
                results["details"].append("AMD SEV-ES (Encrypted State) detected")

        iommu_path = "/sys/kernel/iommu_groups"
        if self._file_exists(iommu_path):
            try:
                iommu_groups = list(Path(iommu_path).iterdir())
                if len(iommu_groups) > 0:
                    results["iommu"] = True
                    results["details"].append(
                        f"IOMMU groups found: {len(iommu_groups)}"
                    )
            except Exception:
                pass

        if cmdline:
            if (
                re.search(
                    r"iommu=pt|iommu=on|intel_iommu=on|amd_iommu=on",
                    cmdline,
                    re.IGNORECASE,
                )
                is not None
            ):
                results["iommu"] = True
                results["details"].append("IOMMU enabled in kernel cmdline")

            if re.search(r"iommu=pt", cmdline, re.IGNORECASE) is not None:
                results["iommu_passthrough"] = True
                results["details"].append("IOMMU passthrough mode in kernel cmdline")

            if (
                re.search(r"maxvfs=|pci=pcie_bus_peer2peer", cmdline, re.IGNORECASE)
                is not None
            ):
                results["sr_iov"] = True
                results["details"].append("SR-IOV parameters in kernel cmdline")

        if self._command_available("lspci"):
            code, output = self._run_command(["lspci", "-nn", "-v"])
            if code == 0:
                if (
                    re.search(r"Single Root I/O Virtualization", output, re.IGNORECASE)
                    is not None
                ):
                    results["sr_iov"] = True
                    results["details"].append("SR-IOV capable device detected")

                if (
                    re.search(r"Access Control Services", output, re.IGNORECASE)
                    is not None
                ):
                    results["details"].append("ACS (Access Control Services) detected")

                if re.search(r"PASID|PRI|ATS", output, re.IGNORECASE) is not None:
                    results["details"].append("PCIe virtualization features detected")

        if self._file_exists("/.dockerenv"):
            results["in_container"] = True
            results["container_type"] = "Docker"
            results["details"].append("Docker container detected (.dockerenv)")

        if cgroup:
            if (
                re.search(r"docker|containerd|kubepods", cgroup, re.IGNORECASE)
                is not None
            ):
                results["in_container"] = True
                if not results["container_type"]:
                    results["container_type"] = "Docker/Container"
                results["details"].append("Container detected in cgroup")

            if re.search(r"kubepods", cgroup, re.IGNORECASE) is not None:
                results["container_type"] = "Kubernetes"
                results["details"].append("Kubernetes container detected")

            if re.search(r"lxc|lxd", cgroup, re.IGNORECASE) is not None:
                if not results["container_type"]:
                    results["container_type"] = "LXC/LXD"
                results["details"].append("LXC/LXD container detected")

        if self._command_available("virt-what"):
            code, output = self._run_command(["virt-what"])
            if code == 0 and output.strip():
                results["in_vm"] = True
                if not results["vm_type"]:
                    results["vm_type"] = output.strip()
                results["details"].append(f"virt-what: {output.strip()}")

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

    memory_info = detector.detect_memory()
    if memory_info["total_gb"] > 0:
        output.append("-" * 60)
        output.append("MEMORY (RAM)")
        output.append("-" * 60)
        output.append(f"Total RAM: {memory_info['total_gb']:.1f} GB")
        output.append(f"Available: {memory_info['available_gb']:.1f} GB")
        output.append(f"Used: {memory_info['used_gb']:.1f} GB")
        if memory_info["swap_gb"] > 0:
            output.append(f"Swap space: {memory_info['swap_gb']:.1f} GB")
        output.append("")
        output.append("What this means:")

        if memory_info["total_gb"] < 8:
            output.append("  - Basic system with limited memory")
            output.append("  - Good for web browsing, documents, light coding")
            output.append("  - Heavy programs (AI, video editing) may be slow")
        elif memory_info["total_gb"] < 16:
            output.append("  - Average system with decent memory")
            output.append("  - Good for most daily tasks, coding, light AI")
            output.append("  - Can handle moderate data processing")
        elif memory_info["total_gb"] < 32:
            output.append("  - Good system with plenty of memory")
            output.append("  - Great for coding, AI, data analysis")
            output.append("  - Can run multiple programs at once")
        else:
            output.append("  - High-end system with lots of memory")
            output.append("  - Excellent for AI, scientific computing")
            output.append("  - Can handle large datasets and complex tasks")

        output.append("")
        output.append("Think of RAM like a desk workspace:")
        output.append(f"  - Your desk is {memory_info['total_gb']:.1f} GB wide")
        output.append(
            f"  - You have {memory_info['available_gb']:.1f} GB free space right now"
        )
        output.append("  - More RAM = more room to spread out your work")
        output.append("  - When RAM fills up, the computer gets slow")

        if memory_info["swap_gb"] > 0:
            output.append("")
            output.append(
                f"  - Swap is like a backup closet ({memory_info['swap_gb']:.1f} GB)"
            )
            output.append("  - When the desk fills, items go to swap")
            output.append("  - Accessing swap is slower than RAM")

        output.append("")

    disk_info = detector.detect_disk()
    output.append("-" * 60)
    output.append("DISK")
    output.append("-" * 60)
    if disk_info["total_gb"] > 0:
        output.append(f"Total Disk Space: {disk_info['total_gb']:.2f} GB")
        output.append(f"Available Space: {disk_info['available_gb']:.2f} GB")
        output.append(f"Used Space: {disk_info['used_gb']:.2f} GB")
        if disk_info["swap_partitions"]:
            total_swap = sum(p["size_gb"] for p in disk_info["swap_partitions"])
            output.append(
                f"Swap partitions available: {len(disk_info['swap_partitions'])} ({total_swap:.2f} GB total)"
            )
        output.append("")
    else:
        output.append("Could not detect disk information")
    output.append("")

    gpu_info = detector.detect_gpu()
    output.append("-" * 60)
    output.append("GRAPHICS PROCESSING UNIT (GPU)")
    output.append("-" * 60)
    if gpu_info["present"]:
        output.append("YES - A graphics accelerator was detected!")
        output.append("")
        if gpu_info.get("make_model"):
            output.append("Hardware identified:")
            for mm in gpu_info["make_model"][:3]:
                output.append(f"  → {mm}")
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

    gpu_details = detector.detect_gpu_device_details()
    if gpu_details["render_devices"] or gpu_details["dri_cards"]:
        output.append("-" * 60)
        output.append("GPU DEVICE DETAILS")
        output.append("-" * 60)

        if gpu_details["is_software_rendering"]:
            output.append("WARNING: Software rendering detected!")
            output.append("  You are using CPU-only rendering (llvmpipe)")
            output.append("  No hardware acceleration available")
            output.append("  This explains slow graphics/AI performance")
            output.append("")

        if gpu_details["render_devices"]:
            output.append(
                f"Found {len(gpu_details['render_devices'])} render device(s):"
            )
            for dev in gpu_details["render_devices"][:3]:
                caps = gpu_details["device_capabilities"].get(dev, {})
                driver = caps.get("driver", "unknown")
                pci_id = caps.get("pci_id", "")
                vendor_name = caps.get("vendor_name", "unknown")
                vendor_desc = caps.get("vendor_description", "")

                output.append(f"  {dev}")
                if vendor_name != "unknown":
                    output.append(f"    Vendor: {vendor_name}")
                if vendor_desc:
                    output.append(f"    Description: {vendor_desc}")
                if driver:
                    output.append(f"    Driver: {driver}")
                if pci_id:
                    output.append(f"    PCI ID: {pci_id}")
            output.append("")

            if gpu_details["vendor_info"]:
                output.append("Vendor ID Reference:")
                for key, value in gpu_details["vendor_info"].items():
                    if key != "glx_vendor":
                        pci_id_part = (
                            key.split(":")[0]
                            if ":" in str(value.get("pci_id", ""))
                            else ""
                        )
                        output.append(
                            f"  {pci_id_part if pci_id_part else 'Unknown'}: {value.get('description', 'Unknown')}"
                        )
                output.append("")

            output.append("What this means:")
            output.append("  - renderD128 is a GPU device for compute/rendering")
            output.append("  - Can be used with OpenCL, Vulkan, OpenGL")
            output.append("  - May be a virtualized GPU (virtio-gpu) if in VM")
            output.append("")

            if gpu_details["rendering_backend"]:
                output.append(f"  OpenGL renderer: {gpu_details['rendering_backend']}")
                if "llvmpipe" in gpu_details["rendering_backend"].lower():
                    output.append(
                        "    (CPU software rendering - no hardware acceleration)"
                    )
                output.append("")

            if gpu_details["vulkan_available"]:
                output.append(
                    "  Vulkan support available - High-performance graphics API"
                )
            if gpu_details["opengl_version"]:
                output.append(f"  OpenGL version: {gpu_details['opengl_version']}")
            output.append("")

            if any(
                "virtio" in str(v).lower() for v in gpu_details["vendor_info"].values()
            ):
                output.append("  Virtio GPU notes:")
                output.append("    - Full CUDA/ROCm may not work in VM")
                output.append("    - Use OpenCL for basic GPU acceleration")
                output.append(
                    "    - Limited compute capabilities compared to bare metal"
                )
                output.append("")
        if gpu_details["dri_cards"]:
            output.append(f"Found {len(gpu_details['dri_cards'])} display device(s):")
            for dev in gpu_details["dri_cards"][:3]:
                output.append(f"  {dev}")
            output.append("")

    npu_info = detector.detect_npu()
    output.append("-" * 60)
    output.append("NEURAL PROCESSING UNIT (NPU) / AI CHIP")
    output.append("-" * 60)
    if npu_info["present"]:
        output.append("YES - An AI accelerator was detected!")
        output.append("")
        if npu_info.get("make_model"):
            output.append("Hardware identified:")
            for mm in npu_info["make_model"][:3]:
                output.append(f"  → {mm}")
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

    software_info = detector.detect_required_software(gpu_info, npu_info)
    if software_info["gpu_software"] or software_info["npu_software"]:
        output.append("-" * 60)
        output.append("SOFTWARE SETUP FOR ACCELERATORS")
        output.append("-" * 60)
        output.append("To use your GPU/NPU with Python, you'll need:")
        output.append("")

        if software_info["drivers"]:
            output.append("1. SYSTEM DRIVERS (install with package manager):")
            for driver in software_info["drivers"][:5]:
                output.append(f"   - {driver}")
            output.append("")

        if software_info["gpu_software"]:
            output.append("2. GPU PYTHON LIBRARIES:")
            for pkg in software_info["gpu_software"][:5]:
                output.append(f"   - {pkg}")
            output.append("")

        if software_info["npu_software"]:
            output.append("3. NPU PYTHON LIBRARIES:")
            for pkg in software_info["npu_software"][:5]:
                output.append(f"   - {pkg}")
            output.append("")

        output.append("Quick start (pick your GPU type):")
        output.append(
            "  NVIDIA:  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )
        output.append(
            "  AMD:     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7"
        )
        output.append("  Intel:   pip install intel_extension_for_pytorch")
        output.append("  Generic: pip install pyopencl")
        output.append("")
        output.append("Note: Install drivers first, then Python libraries!")
        output.append("")

    usage_examples = detector.get_practical_usage_examples(
        gpu_info, npu_info, gpu_details
    )
    if usage_examples["examples"]:
        output.append("-" * 60)
        output.append("PRACTICAL USAGE EXAMPLES")
        output.append("-" * 60)
        for ex in usage_examples["examples"][:30]:
            output.append(ex)
        output.append("")

        if usage_examples["test_commands"]:
            output.append("Quick test commands:")
            for cmd in usage_examples["test_commands"][:5]:
                output.append(cmd)
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

    virt_info = detector.detect_virtualization()
    has_virt_info = any(
        [
            virt_info["in_vm"],
            virt_info["in_container"],
            virt_info["vt_x"],
            virt_info["vt_d"],
            virt_info["iommu"],
            virt_info["sr_iov"],
        ]
    )
    if has_virt_info:
        output.append("-" * 60)
        output.append("VIRTUALIZATION & CONTAINERS")
        output.append("-" * 60)
        if virt_info["in_vm"]:
            vm_type = f" ({virt_info['vm_type']})" if virt_info["vm_type"] else ""
            output.append(f"Running in a Virtual Machine{vm_type}")
            output.append("  This system is a VM, not bare metal hardware")
            output.append("  Performance may be affected by virtualization overhead")
            output.append("")
        if virt_info["in_container"]:
            container_type = (
                f" ({virt_info['container_type']})"
                if virt_info["container_type"]
                else ""
            )
            output.append(f"Running in a Container{container_type}")
            output.append("  This system is running in a containerized environment")
            output.append("  Hardware access may be limited by container isolation")
            output.append("")
        if virt_info["vt_x"] or virt_info["amd_v"]:
            cpu_virt = "Intel VT-x" if virt_info["vt_x"] else "AMD-V"
            output.append(f"{cpu_virt} (CPU Virtualization) - Supported")
            output.append("  This CPU can run virtual machines efficiently")
            output.append("")
        if virt_info["vt_d"] or virt_info["amd_vi"] or virt_info["iommu"]:
            iommu_type = []
            if virt_info["vt_d"]:
                iommu_type.append("Intel VT-d")
            if virt_info["amd_vi"]:
                iommu_type.append("AMD-Vi")
            if not iommu_type:
                iommu_type.append("IOMMU")
            output.append(f"{', '.join(iommu_type)} (I/O Memory Management) - Detected")
            if virt_info["iommu_passthrough"]:
                output.append(
                    "  Passthrough mode: Devices can be passed directly to VMs"
                )
            else:
                output.append("  Hardware can securely assign devices to VMs")
            output.append("")
        if virt_info["sr_iov"]:
            output.append("SR-IOV (Single Root I/O Virtualization) - Available")
            output.append("  Network/storage devices can be shared across multiple VMs")
            output.append("  Each VM gets its own dedicated hardware function")
            output.append("")
        if virt_info["sev"]:
            output.append("AMD SEV (Secure Encrypted Virtualization) - Detected")
            output.append("  VM memory is encrypted for better security")
            if virt_info["sev_es"]:
                output.append("  CPU state is also encrypted (SEV-ES)")
            output.append("")
        if virt_info["details"]:
            output.append("Additional details:")
            for detail in virt_info["details"][:5]:
                output.append(f"  {detail}")
        output.append("")

    virtio_info = detector.detect_virtio_components(virt_info["in_vm"])
    if virtio_info["components"]:
        output.append("-" * 60)
        output.append("VIRTIO COMPONENTS (VM ACCELERATION)")
        output.append("-" * 60)
        output.append("This VM has Virtio components for better performance:")
        output.append("")
        for comp, desc in virtio_info["components"].items():
            output.append(f"{comp}:")
            output.append(f"  {desc}")
        output.append("")
        output.append("What this means:")
        output.append("  - Virtio provides paravirtualized drivers for VMs")
        output.append("  - Faster I/O than traditional emulated hardware")
        output.append("  - Reduces CPU overhead in VM operations")
        output.append("")
        if "virtio-gpu" in virtio_info["components"]:
            output.append("  virtio-gpu: Virtualized GPU (limited compute support)")
            output.append("    - Use OpenCL for basic GPU acceleration")
            output.append("    - Full CUDA/ROCm may not work in VM")
            output.append("")
        if "virtio-net" in virtio_info["components"]:
            output.append("  virtio-net: Optimized network I/O")
            output.append("    - High throughput, low latency networking")
            output.append("")
        if "virtio-blk" in virtio_info["components"]:
            output.append("  virtio-blk: Optimized disk I/O")
            output.append("    - Fast storage access in VM")
            output.append("")

    vfio_info = detector.detect_vfio()
    if vfio_info["available"]:
        output.append("-" * 60)
        output.append("VFIO (VIRTUAL FUNCTION I/O)")
        output.append("-" * 60)
        output.append("VFIO is available for device passthrough to VMs")
        output.append("")

        if vfio_info["modules_loaded"]:
            output.append("VFIO modules loaded:")
            for mod in vfio_info["modules_loaded"]:
                output.append(f"  - {mod}")
            output.append("")

        if vfio_info["iommu_groups"]:
            output.append(f"IOMMU groups: {len(vfio_info['iommu_groups'])}")
            for group_num in sorted(vfio_info["iommu_groups"][:5]):
                devices = vfio_info["iommu_devices"].get(group_num, [])
                output.append(f"  Group {group_num}: {', '.join(devices[:3])}")
            output.append("")

        if vfio_info["devices_bound_to_vfio"]:
            output.append("Devices bound to VFIO (can be passed to VMs):")
            for dev in vfio_info["devices_bound_to_vfio"][:5]:
                output.append(f"  {dev}")
            output.append("")

        if vfio_info["vfio_devices"]:
            output.append(f"VFIO device nodes: {len(vfio_info['vfio_devices'])}")
            for dev in vfio_info["vfio_devices"][:3]:
                output.append(f"  {dev}")
            output.append("")

        if vfio_info["mdev_devices"]:
            output.append("Mediated devices (shared device assignment):")
            for mdev in vfio_info["mdev_devices"][:3]:
                output.append(f"  {mdev}")
            output.append("")

        if vfio_info["platform_devices"]:
            output.append("VFIO platform devices:")
            for plat in vfio_info["platform_devices"][:3]:
                output.append(f"  {plat}")
            output.append("")

        output.append("What this means:")
        output.append(
            "  - PCI devices can be passed directly to VMs with near-native performance"
        )
        output.append("  - Devices bound to VFIO bypass host kernel drivers")
        output.append("  - Requires IOMMU for secure device assignment")
        output.append("  - Use for GPU, NIC, storage passthrough to VMs")
        output.append("")

        if vfio_info["details"]:
            output.append("Additional details:")
            for detail in vfio_info["details"][:5]:
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
    if virt_info["vt_x"] or virt_info["amd_v"]:
        accelerators.append("CPU Virtualization (VT-x/AMD-V)")
    if virt_info["iommu"]:
        accelerators.append("IOMMU (VT-d/AMD-Vi)")
    if virt_info["sr_iov"]:
        accelerators.append("SR-IOV")
    if virt_info["sev"] or virt_info["sev_es"]:
        accelerators.append("Encrypted Virtualization (SEV)")
    if vfio_info["available"]:
        accelerators.append("VFIO (Device Passthrough)")

    if accelerators:
        output.append("Your system has these acceleration features:")
        for acc in accelerators:
            output.append(f"  {acc}")
        output.append("")

        if gpu_info.get("make_model"):
            output.append("GPU details:")
            for mm in gpu_info["make_model"][:2]:
                output.append(f"  {mm}")
            output.append("")

        if npu_info.get("make_model"):
            output.append("NPU details:")
            for mm in npu_info["make_model"][:2]:
                output.append(f"  {mm}")
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
