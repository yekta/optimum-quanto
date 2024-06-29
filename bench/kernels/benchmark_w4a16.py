# From: https://github.com/IST-DASLab/marlin/blob/master/bench.py
import time

import torch

from optimum.quanto import AWQPackedTensor, AWQPacking, MarlinPackedTensor


# FIXME: should implement permute_scales() inside MarlinBitsTensor
def _get_scale_perm():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


_scale_perm, _ = _get_scale_perm()


def permute_scales(scales: torch.Tensor):
    s = scales
    _, N = s.shape
    s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
    s = s.reshape((-1, N)).contiguous()
    return s


def benchmark(f, warmup=1, iter=10):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.0)
    return res


def get_problem(m, n, k, groupsize=128):
    dev = torch.device("cuda:0")
    A = torch.rand((m, k), dtype=torch.half, device=dev)
    B_4bit = torch.randint(0, 2**4, (n, k), dtype=torch.uint8, device=dev)
    B_awq = AWQPackedTensor.pack(B_4bit, packing=AWQPacking.V2)._data
    B_marlin = MarlinPackedTensor.pack(B_4bit)._data
    B_ref = torch.rand((k, n), dtype=torch.half, device=dev)
    s = torch.rand((k // groupsize, n), dtype=torch.half, device=dev) / 2**4
    s_marlin = permute_scales(s)
    z = torch.randint(-(2 ** (4 - 1)), 2 ** (4 - 1), (k // groupsize, n), dtype=torch.int8, device=dev)
    sz = -z * s
    sz_marlin = permute_scales(sz)
    torch.cuda.synchronize()
    return A, B_ref, B_awq, B_marlin, s, s_marlin, sz, sz_marlin


def benchmark_dense(A, B, m, n, k):
    res = benchmark(lambda: torch.matmul(A, B))
    return {
        "s": res,
        "TFLOP/s": 2 * A.numel() * n / res / 10**12,
        "GB/s": (2 * A.numel() + 2 * B.numel() + 2 * (m * n)) / res / 10**9,
    }


def benchmark_awq(A, B, s, sz, m, n, k):
    res = benchmark(lambda: torch.ops.quanto.gemm(A, B, s, sz, rows=m, out_cols=n, in_cols=k, bits=4, group_size=128))
    return {
        "s": res,
        "TFLOP/s": 2 * (m * k) * n / res / 10**12,
        "GB/s": (2 * A.numel() + 2 * B.numel() + 2 * (m * n) + 2 * s.numel() + 2 * sz.numel()) / res / 10**9,
    }


def benchmark_marlin(A, B, s, sz, m, n, k):
    workspace = torch.zeros(n // 128 * 16, dtype=torch.int, device=torch.device("cuda:0"))
    res = benchmark(lambda: torch.ops.quanto.gemm_marlin(A, B, s, sz, workspace))
    return {
        "s": res,
        "TFLOP/s": 2 * (m * k) * n / res / 10**12,
        "GB/s": (2 * A.numel() + 4 * B.numel() + 2 * (m * n) + 2 * s.numel() + 2 * sz.numel()) / res / 10**9,
    }


MODELS = {
    "Llama7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
    "Llama13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
    "Llama33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
    "Llama65B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
    "Falcon180B": [
        # Note that parallel attention and FC allows layer fusions
        (14848, 14848 * 5 + 1024),
        (14848 * 5, 14848),
    ],
}

# Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments
ALL = False

for groupsize in [-1, 128] if ALL else [128]:
    print("groupsize=%d" % groupsize)
    print()
    for model, layers in MODELS.items():
        print(model)
        if ALL:
            batchsizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        else:
            batchsizes = [1, 2, 4, 8, 16, 32, 64, 128]
        for batch in batchsizes:
            if not ALL and batch != 16:
                continue
            tot_awq = {"s": 0, "TFLOP/s": 0, "GB/s": 0, "speedup": 0}
            tot_marlin = {"s": 0, "TFLOP/s": 0, "GB/s": 0, "speedup": 0}
            for layer in layers:
                A, B_ref, B_awq, B_marlin, s, s_marlin, sz, sz_marlin = get_problem(
                    batch, layer[1], layer[0], groupsize
                )
                res_d = benchmark_dense(A, B_ref, batch, layer[1], layer[0])
                res_awq = benchmark_awq(A, B_awq, s, sz, batch, layer[1], layer[0])
                res_marlin = benchmark_marlin(A, B_marlin, s_marlin, sz_marlin, batch, layer[1], layer[0])
                res_awq["speedup"] = res_d["s"] / res_awq["s"]
                res_marlin["speedup"] = res_d["s"] / res_marlin["s"]
                tot_awq["s"] += res_awq["s"]
                tot_marlin["s"] += res_marlin["s"]
                for k in tot_awq:
                    if k != "s":
                        tot_awq[k] += res_awq[k] * res_awq["s"]
                for k in tot_marlin:
                    if k != "s":
                        tot_marlin[k] += res_marlin[k] * res_marlin["s"]
            for k in tot_awq:
                if k != "s":
                    tot_awq[k] /= tot_awq["s"]
            for k in tot_marlin:
                if k != "s":
                    tot_marlin[k] /= tot_marlin["s"]
            print(
                "AWQ, batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f"
                % (batch, tot_awq["s"], tot_awq["TFLOP/s"], tot_awq["GB/s"], tot_awq["speedup"])
            )
            print(
                "Marlin, batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f"
                % (batch, tot_marlin["s"], tot_marlin["TFLOP/s"], tot_marlin["GB/s"], tot_marlin["speedup"])
            )
        print()
