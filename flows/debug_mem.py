import os

import psutil


def get_memory_and_cpu_metrics():
    """Get CPU and memory metrics for debugging"""
    process = psutil.Process(os.getpid())

    # CPU info
    cpu_percent = process.cpu_percent()

    # Memory info
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()

    # System memory
    system_memory = psutil.virtual_memory()

    return {
        "cpu_percent": cpu_percent,
        "memory_mb": memory_info.rss / 1024 / 1024,
        "memory_percent": memory_percent,
        "system_memory_percent": system_memory.percent,
        "system_memory_available_gb": system_memory.available / 1024 / 1024 / 1024,
    }
