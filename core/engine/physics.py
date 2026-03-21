#type: ignore
import taichi as ti

# Initialize Taichi to automatically use Metal on Mac
try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)

@ti.kernel
def compute_risk_kernel(
    padded_pdf: ti.types.ndarray(dtype=ti.f32),
    sm_premult: ti.types.ndarray(dtype=ti.f32),
    out_map: ti.types.ndarray(dtype=ti.f32),
    pcy: int,
    pcx: int
):
    nr = out_map.shape[0]
    nc = out_map.shape[1]

    for y, x in out_map:
        acc = 0.0
        for r in range(nr):
            for c in range(nc):
                val = padded_pdf[(pcy - y) + r, (pcx - x) + c]
                acc += val * sm_premult[r, c]
        
        out_map[y, x] = acc