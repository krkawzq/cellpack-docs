"""External sample ordering with fixed accesses or a native AccessFn.

uv run example/dataloader-example.py
uv run example/dataloader-example.py --access-fn
uv run example/dataloader-example.py --device cuda

Torch is installed separately. A C++17 compiler builds the model template
and the optional access function; sampling policy is ordinary caller code.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pycellpack import write
from pycellpack.dataloader import (
    Accesses,
    AccessFn,
    ScDataLoader,
    ScDataset,
    process_presets,
)

ACCESS_CODE = r'''
#include "access_fn_cabi.h"
struct Visits { const uint32_t *datasets; const uint64_t *rows; };
extern "C" int32_t generate_accesses(void *context, uint64_t start, uint64_t count,
                                     uint32_t *datasets, uint64_t *rows) {
    const auto &visits = *static_cast<const Visits *>(context);
    for (uint64_t i = 0; i < count; ++i) {
        datasets[i] = visits.datasets[start + i];
        rows[i] = visits.rows[start + i];
    }
    return 0;
}
'''


def sample_order(corpus, rng):
    counts = [len(dataset) for dataset in corpus]
    datasets = np.repeat(np.arange(len(corpus), dtype=np.uint32), counts)
    rows = np.concatenate([np.arange(n, dtype=np.uint64) for n in counts])
    order = rng.permutation(len(rows))
    return Accesses(datasets[order], rows[order])


def bind_access_fn(accesses):
    import ctypes

    class Visits(ctypes.Structure):
        _fields_ = [("datasets", ctypes.c_void_p), ("rows", ctypes.c_void_p)]

    tables = Visits(accesses.dataset_ids.ctypes.data, accesses.rows.ctypes.data)
    return AccessFn.from_source(code=ACCESS_CODE, kind="corpus", length=len(accesses),
                                user_data=tables, key=(accesses.dataset_ids, accesses.rows), owner=accesses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--access-fn", action="store_true")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="pycellpack-dataloader-") as directory:
        root = Path(directory)
        rng = np.random.default_rng(0)
        a = rng.poisson(2, (7, 6)).astype(np.float32)
        b = rng.poisson(2, (5, 3)).astype(np.float32)
        a[:, 0] += 1
        b[:, 0] += 1
        corpus = [
            ScDataset(write(root / "a.cpk", a, workers=1), ids=np.arange(6)),
            ScDataset(write(root / "b.cpk", b, workers=1), ids=np.array([0, 2, 4])),
        ]
        process = process_presets.scgpt(np.arange(10, 16), pad_token=0, cls_token=1, length=8, seed=3)
        accesses = sample_order(corpus, rng)
        inputs = {"access_fn": bind_access_fn(accesses)} if args.access_fn else {"accesses": accesses}
        with ScDataLoader(corpus=corpus, out_ids=np.arange(6), process=process, batch_size=4,
                          workers=2, prefetch=2, device=args.device,
                          contiguous=("genes", "values"), **inputs) as loading:
            print(loading)
            for epoch in range(2):
                accesses = sample_order(corpus, rng)
                if args.access_fn:
                    loading.access_fn = bind_access_fn(accesses)
                else:
                    loading.accesses = accesses
                for step, batch in enumerate(loading):
                    print(f"epoch={epoch} step={step} " + ", ".join(
                        f"{name}={tuple(tensor.shape)}@{tensor.device}" for name, tensor in batch.items()
                    ))
                print(f"accesses={loading.stats.accesses}")


if __name__ == "__main__":
    main()
