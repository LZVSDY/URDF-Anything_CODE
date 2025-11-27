import os
import torch

script_path = os.path.dirname(os.path.abspath(__file__))
# from torch.utils.cpp_extension import _get_cuda_arch_flags
# # Hack: add sm_90 support for H800
# original_get_arch_flags = _get_cuda_arch_flags

# def patched_get_cuda_arch_flags():
#     flags = original_get_arch_flags()
#     # Add sm_90 if not present
#     if '90' not in str(flags):
#         flags.append('9.0')  # or '10.0' depending on PyTorch version
#     return flags

# import torch.utils.cpp_extension
# torch.utils.cpp_extension._get_cuda_arch_flags = patched_get_cuda_arch_flags
from torch.utils.cpp_extension import load

# if torch.cuda.is_available():
#     cd = load(name="cd",
#               sources=[os.path.join(script_path, "chamfer_distance.cpp"),
#                        os.path.join(script_path, "chamfer_distance.cu")])


class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1

    @staticmethod
    def backward(ctx, graddist1, graddist2, _):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


# class ChamferDistance(torch.nn.Module):
#     def forward(self, xyz1, xyz2):
#         return ChamferDistanceFunction.apply(xyz1, xyz2)

# ===== 纯 PyTorch 实现 =====
def chamfer_distance_pytorch(xyz1, xyz2):
    diff = xyz1.unsqueeze(2) - xyz2.unsqueeze(1)   # [B, N, M, 3]
    dist_table = torch.sum(diff ** 2, dim=-1)      # [B, N, M]
    dist1, idx1 = torch.min(dist_table, dim=2)     # [B, N]
    dist2, idx2 = torch.min(dist_table, dim=1)     # [B, M]
    return dist1, dist2, idx1

class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return chamfer_distance_pytorch.apply(xyz1, xyz2)
