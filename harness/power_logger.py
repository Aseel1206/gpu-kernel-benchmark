"""Background power/temperature sampler, used during bench.py's timed loop.

Tries pynvml first (lower overhead, in-process); falls back to polling `nvidia-smi` in a
subprocess if pynvml isn't installed. On a platform with no NVIDIA GPU (e.g. this dev machine),
both backends are unavailable and PowerLogger becomes a no-op so the harness still runs end to end.
"""
import subprocess
import threading
import time


class PowerLogger:
    def __init__(self, gpu_index: int = 0, poll_interval_s: float = 0.05):
        self.gpu_index = gpu_index
        self.poll_interval_s = poll_interval_s
        self._thread = None
        self._stop_flag = threading.Event()
        self._samples_power = []
        self._samples_temp = []
        self.avg_power_w = None
        self.peak_power_w = None
        self.avg_temp_c = None
        self._backend = self._detect_backend()

    def _detect_backend(self):
        try:
            import pynvml  # noqa: F401
            return "pynvml"
        except ImportError:
            pass
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, check=True, timeout=5)
            return "nvidia-smi"
        except Exception:
            return None

    def _sample_pynvml(self):
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        while not self._stop_flag.is_set():
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            self._samples_power.append(power_w)
            self._samples_temp.append(temp_c)
            time.sleep(self.poll_interval_s)
        pynvml.nvmlShutdown()

    def _sample_nvidia_smi(self):
        while not self._stop_flag.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     f"--id={self.gpu_index}",
                     "--query-gpu=power.draw,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                power_str, temp_str = out.split(",")
                self._samples_power.append(float(power_str))
                self._samples_temp.append(float(temp_str))
            except Exception:
                pass
            time.sleep(self.poll_interval_s)

    def start(self):
        if self._backend is None:
            return  # no-op: no NVIDIA GPU/driver available on this machine
        self._samples_power = []
        self._samples_temp = []
        self._stop_flag.clear()
        target = self._sample_pynvml if self._backend == "pynvml" else self._sample_nvidia_smi
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        if self._backend is None:
            return
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._samples_power:
            self.avg_power_w = sum(self._samples_power) / len(self._samples_power)
            self.peak_power_w = max(self._samples_power)
        if self._samples_temp:
            self.avg_temp_c = sum(self._samples_temp) / len(self._samples_temp)
