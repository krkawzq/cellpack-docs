"""Inspect scGPT output on CPU or transfer one packed batch to CUDA.

uv run example/processfn-example.py                 # CPU field views
uv run example/processfn-example.py --device cuda   # packed H2D, then GPU fields

Requires pycellpack and a C++17 compiler; CUDA mode also requires PyTorch
with CUDA. ProcessFn writes one packed record per row in both modes.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pycellpack import DataSource, Loader, write
from pycellpack.loader import Ring
from pycellpack.loader.process_fn import presets


def show_cpu(process, raw):
    # These views retain the slot only until this function returns.
    for name, array in process.unpack(raw).items():
        print(f"{name}: {array}")


def upload(batch, device, copy_stream):
    import torch

    raw = batch.region("scgpt")
    host = torch.from_numpy(raw)
    try:
        with torch.cuda.stream(copy_stream):
            packed = host.to(device, non_blocking=True)
    finally:
        # Keep both host views alive until this stream finishes reading the slot.
        copy_stream.synchronize()
    batch.release()
    del host, raw
    return packed


def gpu_fields(packed, output, streams):
    import torch

    current = torch.cuda.current_stream(packed.device)
    fields = {}
    dtypes = {"genes": torch.int64, "values": torch.float32, "targets": torch.float32}
    for name, dtype in dtypes.items():
        stream = streams[name]
        with torch.cuda.stream(stream):
            packed.record_stream(stream)
            offset = output.offsets[name]
            width = output.dtype[name].itemsize
            view = packed[:, offset:offset + width].view(dtype)
            # Each field's contiguous copy can run on its own CUDA stream.
            fields[name] = view.contiguous()
        current.wait_stream(stream)
        fields[name].record_stream(current)
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    counts = np.array([[0, 2, 0, 4, 1, 0], [3, 0, 1, 0, 2, 1]], np.float32)
    out_ids = np.arange(counts.shape[1])
    process = presets.scgpt(
        tokens=[10, 11, 12, 13, 14, -1],
        pad_token=0,
        cls_token=1,
        length=8,
        mask_ratio=0.4,
        seed=7,
    )
    print(process)
    print(process.output)
    with TemporaryDirectory(prefix="pycellpack-") as directory:
        store = write(Path(directory) / "counts.cpk", counts, format="csr", workers=1)
        source = DataSource([(store, out_ids)], len(out_ids))
        loading = Loader(source, process.schema(out_ids, name="scgpt"), workers=2, slot_rows=2)
        print(loading)
        ring = None
        if args.device == "cuda":
            import torch

            device = torch.device("cuda:0")
            copy_stream = torch.cuda.Stream(device=device)
            streams = {name: torch.cuda.Stream(device=device) for name in process.output.names}
            layout = loading.ring_layout(slots=2)
            storage = torch.empty(layout.data_bytes, dtype=torch.uint8, pin_memory=True)
            ring = Ring(layout, data=storage.numpy())
        with loading.start([1, 0], ring=ring) as run:
            for batch in run:
                print(batch)
                with batch:
                    if args.device == "cpu":
                        show_cpu(process, batch.region("scgpt"))
                        continue
                    packed = upload(batch, device, copy_stream)
                # Host views are gone; only device storage is used from here.
                fields = gpu_fields(packed, process.output, streams)
                for name, tensor in fields.items():
                    print(f"{name}: {tensor}")
            run.join()
        print(run)


if __name__ == "__main__":
    main()
