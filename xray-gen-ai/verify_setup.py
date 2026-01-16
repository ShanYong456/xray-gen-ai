import sys
import torch
from gvxrPython3 import gvxr


def check_setup():
    # 1. Python version
    print(f"Python Version: {sys.version.split()[0]}")

    # 2. Check Deep Learning GPU
    if torch.cuda.is_available():
        print(f"⬛  PyTorch GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("+ PyTorch cannot find GPU!")

    # 3. Check Simulation Engine
    try:
        gvxr.createOpenGLContext()
        print("⬛  gVirtualXray: OpenGL Context Created Successfully")

        # Simple test: Create a source
        gvxr.setSourcePosition(0, 0, 0, "cm")
        print("⬛  gVirtualXray: Simulation Logic Active")
    except Exception as e:
        print(f"+ gVirtualXray Error: {e}")
        print("  (If on WSL, ensure WSLg is active)")


if __name__ == "__main__":
    check_setup()