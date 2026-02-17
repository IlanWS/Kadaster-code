"""Setup script using requirements.txt."""

import subprocess
from pathlib import Path

from setuptools import find_packages, setup


def detect_rocm_version():
    """Detect ROCm version by checking librocm-core.so.x version."""
    try:
        result = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "librocm-core" in line and ".so" in line:
                import re

                match = re.search(r"librocm-core\.so\.(\d+)", line)
                if match:
                    return True
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def get_requirements():
    """Get requirements, optionally adding ROCm packages for AMD GPUs."""
    base_path = Path(__file__).parent
    req_file = base_path / "requirements.txt"
    requirements = req_file.read_text().splitlines()

    filtered = []

    for req in requirements:
        stripped = req.strip()
        if not stripped or stripped.startswith("-e") or stripped.startswith("#"):
            continue

        lower_req = stripped.lower()
        if "tensorflow" in lower_req or "keras" in lower_req:
            continue

        filtered.append(stripped)

    return filtered


def main():
    install_requires = get_requirements()

    rocm_available = detect_rocm_version()

    if rocm_available:
        print(
            f"ROCm detected. Adding ROCm-compatible TensorFlow packages."
        )
        import sys

        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        rocm_repo = f"https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.4/"
        install_requires.extend(
            [
                f"tensorflow-rocm @ {rocm_repo}tensorflow_rocm-2.18.1-{python_version}-{python_version}-manylinux_2_28_x86_64.whl",
                "keras>=3.0.0",
            ]
        )
    else:
        print(f"No ROCm detected.")
        install_requires.append("tensorflow>=2.0.0")

    setup(
        name="kadaster-code",
        version="0.1.0",
        description="Code for processing Kadaster/BRT data and downloading images",
        python_requires="==3.12.*",
        install_requires=install_requires,
        packages=find_packages(),
    )


if __name__ == "__main__":
    main()
