import torch
from cumem_allocator import CuMemAllocator

allocator = CuMemAllocator.get_instance()

with allocator.use_memory_pool():
    a = torch.randn(10, 10, device='cuda')

b = a.clone()
torch.cuda.synchronize()

allocator.sleep()
print(f"finish sleep")
allocator.wake_up()
print(f"finish wake up")

assert torch.equal(a, b), "Tensors are not equal!"
