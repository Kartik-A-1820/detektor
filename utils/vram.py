import torch

def set_vram_cap(fraction=0.8):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)

def vram_report(prefix=""):
    if not torch.cuda.is_available():
        return "CUDA not available"
    alloc = torch.cuda.memory_allocated()/1024**2
    total = torch.cuda.get_device_properties(0).total_memory/1024**2
    return f"{prefix}Allocated: {alloc:.1f}MB / {total:.1f}MB"
