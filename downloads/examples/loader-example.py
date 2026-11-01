"""Illustrative scGPT loader: three equivalent native ProcessFn bindings.

Read this as example code, not as a script or an importable module. Ellipses
stand for prepared data. The C++ and Numba implementations below independently
produce the same packed scGPT output (model=0).
Inputs are canonical CSR rows or dense float32 rows of finite counts. CSR
columns are unique but need not be sorted. Choose one binding method.
"""

import ctypes as C
import inspect

import numpy as np
import torch
from numba import cfunc, njit
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerModel

from pycellpack.loader import (
    DataSource,
    Loader,
    OutputSchema,
    ProcessFn,
    Ring,
)
from pycellpack.loader.process_fn import numba_bindings

cpk_path = "data/my-dataset/raw.cpk"
gene_tokens = ...  # Read-only int64 array: token per column; -1 excludes a column
vocab = ...  # Prepared scGPT vocabulary, including <pad> and <cls>
accesses = ...  # uint32 cell IDs in training access order
length, batch_size, slots = 1024, 32, 4


# Match the full Config ABI so the same configuration also works with the
# existing sampling.so. Only the scGPT fields are used by this example.
class Config(C.Structure):
    _fields_ = [
        ("seed", C.c_uint64),
        *[
            (k, C.c_uint32)
            for k in (
                "model",
                "length",
                "sample",
                "positive",
                "negative",
                "shared",
                "global_size",
                "batch",
            )
        ],
        *[(k, C.c_int64) for k in ("pad", "cls", "chrom_offset")],
        ("mask", C.c_double),
        *[(k, C.c_uint32) for k in ("panel", "dataset", "raw_counts", "stream")],
        *[(k, C.c_void_p) for k in ("tokens", "chroms", "starts")],
        ("n_valid", C.c_uint32),
        ("valid", C.c_void_p),
        ("reverse", C.c_void_p),
        ("local", C.c_void_p),
    ]


CONFIG_DTYPE = np.dtype(
    [
        ("seed", np.uint64),
        *[
            (k, np.uint32)
            for k in (
                "model",
                "length",
                "sample",
                "positive",
                "negative",
                "shared",
                "global_size",
                "batch",
            )
        ],
        *[(k, np.int64) for k in ("pad", "cls", "chrom_offset")],
        ("mask", np.float64),
        *[(k, np.uint32) for k in ("panel", "dataset", "raw_counts", "stream")],
        *[(k, np.uint64) for k in ("tokens", "chroms", "starts")],
        ("n_valid", np.uint32),
        ("valid", np.uint64),
        ("reverse", np.uint64),
        ("local", np.uint64),
    ],
    align=True,
)

params = Config(
    seed=0,
    model=0,
    length=length,
    batch=batch_size,
    pad=vocab["<pad>"],
    cls=vocab["<cls>"],
    mask=0.4,
    panel=len(gene_tokens),
    tokens=gene_tokens.ctypes.data,
)
parameter_key = (
    params.seed,
    params.length,
    params.pad,
    params.cls,
    params.mask,
    gene_tokens,
)
binding = {"user_data": params, "key": parameter_key, "owner": (gene_tokens,)}


# Method 1: bind the existing compiled provider's exported symbols.
def process_from_so():
    return ProcessFn.from_library(
        "./sampling.so", csr="sampling_csr", dense="sampling_dense", **binding
    )


# Method 2: pass actual C++ source to pycellpack for compilation and caching.
# Floating-point contraction is disabled to match the Numba implementation's
# separate quantile interpolation operations; neither implementation uses fastmath.
CPP_SOURCE = r"""
#include "process_fn_cabi.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

struct Config {
    uint64_t seed;
    uint32_t model, length, sample, positive, negative, shared, global_size, batch;
    int64_t pad, cls, chrom_offset;
    double mask;
    uint32_t panel, dataset, raw_counts, stream;
    const int64_t *tokens, *chroms, *starts;
    uint32_t n_valid;
    const uint32_t *valid;
    const int64_t *reverse;
    const uint32_t *local;
};
static uint64_t mix(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
struct Rng {
    uint64_t state;
    uint64_t next() { return mix(state += UINT64_C(0x9e3779b97f4a7c15)); }
    uint64_t below(uint64_t n) {
        uint64_t x = next();
        if (x < n) {
            const uint64_t threshold = -n % n;
            while (x < threshold) x = next();
        }
        return x % n;
    }
    double unit() { return (next() >> 11) * 0x1.0p-53; }
};
struct Entry { uint32_t gene; double value; uint64_t tie; };

static int scgpt(const PyCellPackContext &context, uint64_t dim,
                 std::vector<Entry> &entries, uint8_t *out) {
    const auto &c = *static_cast<const Config *>(context.user_data);
    if (context.dtype != 8 || dim != c.panel) return -10;
    if (c.model != 0) return -11;
    std::sort(entries.begin(), entries.end(), [](auto &a, auto &b) { return a.gene < b.gene; });
    entries.erase(std::remove_if(entries.begin(), entries.end(),
        [&](auto &x) { return c.tokens[x.gene] < 0; }), entries.end());
    if (entries.empty()) return -20;

    auto *ids = reinterpret_cast<int64_t *>(out);
    auto *values = reinterpret_cast<float *>(ids + c.length);
    auto *target = values + c.length;
    std::fill(ids, ids + c.length, c.pad);
    std::fill(values, values + c.length, -2.f);
    std::fill(target, target + c.length, -2.f);
    ids[0] = c.cls;

    std::vector<double> sorted;
    for (auto &x : entries) sorted.push_back(x.value);
    std::sort(sorted.begin(), sorted.end());
    double bins[50];
    for (int i = 0; i < 50; ++i) {
        double p = (sorted.size() - 1) * (double(i) / 49);
        size_t j = size_t(p);
        bins[i] = sorted[j] + (sorted[std::min(j + 1, sorted.size() - 1)] - sorted[j]) * (p - j);
    }
    Rng r{mix(c.seed ^ mix(context.cell + UINT64_C(0x63656c6c)) ^ mix(c.model))};
    // Every eligible gene consumes a tie draw before subsampling.
    for (auto &x : entries) x.tie = r.next();
    size_t n = std::min(entries.size(), size_t(c.length - 1));
    std::vector<uint32_t> selected;
    if (entries.size() > c.length - 1) {
        std::vector<uint8_t> picked(entries.size(), 0);
        for (size_t j = entries.size() - n; j < entries.size(); ++j) {
            uint32_t v = r.below(j + 1);
            if (picked[v]) v = j;
            picked[v] = 1;
            selected.push_back(v);
        }
        for (size_t i = selected.size(); i > 1; --i)
            std::swap(selected[i - 1], selected[r.below(i)]);
    }
    for (size_t i = 0; i < n; ++i) {
        const auto &x = entries[selected.empty() ? i : selected[i]];
        int left = std::upper_bound(bins, bins + 50, x.value) - bins;
        int right = std::lower_bound(bins, bins + 50, x.value) - bins;
        double binned = std::ceil(((x.tie >> 11) * 0x1.0p-53) * (right - left) + left);
        ids[i + 1] = c.tokens[x.gene];
        target[i + 1] = binned;
        values[i + 1] = r.unit() < c.mask ? -1.f : float(binned);
    }
    return 0;
}
extern "C" int32_t sampling_csr(const PyCellPackCsrRow *row) {
    try {
        if (row->nnz && (!row->columns || !row->values)) return -12;
        const auto *values = static_cast<const float *>(row->values);
        std::vector<Entry> entries;
        for (uint64_t i = 0; i < row->nnz; ++i)
            if (values[i] > 0) entries.push_back({row->columns[i], values[i], 0});
        return scgpt(row->context, row->out_dim, entries, row->out);
    } catch (...) { return -99; }
}
extern "C" int32_t sampling_dense(const PyCellPackDenseRow *row) {
    try {
        const auto *values = static_cast<const float *>(row->values);
        std::vector<Entry> entries;
        for (uint64_t i = 0; i < row->out_dim; ++i)
            if (values[i] > 0) entries.push_back({uint32_t(i), values[i], 0});
        return scgpt(row->context, row->out_dim, entries, row->out);
    } catch (...) { return -99; }
}
"""


def process_from_source():
    return ProcessFn.from_source(
        code=CPP_SOURCE,
        language="c++",
        std="c++17",
        flags=["-ffp-contract=off"],
        csr="sampling_csr",
        dense="sampling_dense",
        **binding,
    )


# Method 3: independently implement the same algorithm in Numba.
# uint64 arithmetic, RNG draw order, sorted gene order, quantile interpolation,
# tie handling, Floyd sampling, shuffling, masking, and padding are explicit.
pn = numba_bindings()


@njit
def numba_mix(x):
    x = np.uint64(x)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit
def numba_next(state):
    state = state + np.uint64(0x9E3779B97F4A7C15)
    return state, numba_mix(state)


@njit
def numba_below(state, bound):
    bound = np.uint64(bound)
    state, x = numba_next(state)
    if x < bound:
        threshold = (np.uint64(0) - bound) % bound
        while x < threshold:
            state, x = numba_next(state)
    return state, np.int64(x % bound)


@njit
def numba_scgpt(config, cell, columns, raw_values, out):
    tokens = pn.array_at(config.tokens, config.panel, np.int64)
    genes = np.empty(len(columns), dtype=np.uint32)
    values = np.empty(len(columns), dtype=np.float64)
    count = 0
    for i in range(len(columns)):
        if raw_values[i] > 0 and tokens[columns[i]] >= 0:
            genes[count] = columns[i]
            values[count] = raw_values[i]
            count += 1
    if count == 0:
        return -20
    # Sorting by source column makes replay independent of codec column order.
    order = np.argsort(genes[:count])
    genes = genes[:count][order]
    values = values[:count][order]
    sorted_values = np.sort(values)
    bins = np.empty(50, dtype=np.float64)
    for i in range(50):
        p = (count - 1) * (i / 49.0)
        j = int(p)
        bins[i] = sorted_values[j] + (
            sorted_values[min(j + 1, count - 1)] - sorted_values[j]
        ) * (p - j)

    length = np.int64(config.length)
    ids = out[: length * 8].view(np.int64)
    masked_values = out[length * 8 : length * 12].view(np.float32)
    target = out[length * 12 :].view(np.float32)
    ids[:] = config.pad
    masked_values[:] = -2.0
    target[:] = -2.0
    ids[0] = config.cls

    state = numba_mix(
        config.seed
        ^ numba_mix(np.uint64(cell) + np.uint64(0x63656C6C))
        ^ numba_mix(np.uint64(config.model))
    )
    ties = np.empty(count, dtype=np.uint64)
    for i in range(count):
        state, ties[i] = numba_next(state)
    n = min(count, length - 1)
    selected = np.empty(n, dtype=np.int64)
    if count > length - 1:
        picked = np.zeros(count, dtype=np.bool_)
        for i in range(n):
            j = count - n + i
            state, v = numba_below(state, j + 1)
            if picked[v]:
                v = j
            picked[v] = True
            selected[i] = v
        for i in range(n, 1, -1):
            state, j = numba_below(state, i)
            selected[i - 1], selected[j] = selected[j], selected[i - 1]
    else:
        for i in range(n):
            selected[i] = i
    for i in range(n):
        j = selected[i]
        left = np.searchsorted(bins, values[j], side="right")
        right = np.searchsorted(bins, values[j], side="left")
        unit = float(ties[j] >> np.uint64(11)) * (2.0**-53)
        binned = np.ceil(unit * (right - left) + left)
        ids[i + 1] = tokens[genes[j]]
        target[i + 1] = binned
        state, draw = numba_next(state)
        masked_values[i + 1] = (
            -1.0 if float(draw >> np.uint64(11)) * (2.0**-53) < config.mask else binned
        )
    return 0


@cfunc(pn.CSR_SIGNATURE)
def csr_callback(call):
    row = pn.row(call)
    if row.nnz != 0 and (row.columns == 0 or row.values == 0):
        return -12
    config = pn.array_at(row.context.user_data, 1, CONFIG_DTYPE)[0]
    if row.context.dtype != 8 or row.out_dim != config.panel:
        return -10
    if config.model != 0:
        return -11
    columns = pn.array_at(row.columns, row.nnz, np.uint32)
    values = pn.array_at(row.values, row.nnz, np.float32)
    out = pn.array_at(row.out, np.int64(config.length) * 16, np.uint8)
    return numba_scgpt(config, row.context.cell, columns, values, out)


@cfunc(pn.DENSE_SIGNATURE)
def dense_callback(call):
    row = pn.row(call)
    config = pn.array_at(row.context.user_data, 1, CONFIG_DTYPE)[0]
    if row.context.dtype != 8 or row.out_dim != config.panel:
        return -10
    if config.model != 0:
        return -11
    columns = np.arange(row.out_dim, dtype=np.uint32)
    values = pn.array_at(row.values, row.out_dim, np.float32)
    out = pn.array_at(row.out, np.int64(config.length) * 16, np.uint8)
    return numba_scgpt(config, row.context.cell, columns, values, out)


def process_from_cfunc():
    # Include helper implementations and the global record layout in identity,
    # because the callback bodies delegate to these Numba helpers.
    helpers = (numba_mix, numba_next, numba_below, numba_scgpt)
    key = (
        parameter_key,
        CONFIG_DTYPE.descr,
        tuple(inspect.getsource(helper.py_func) for helper in helpers),
    )
    return ProcessFn.from_cfunc(
        csr=csr_callback,
        dense=dense_callback,
        user_data=params,
        key=key,
        owner=(gene_tokens,),
    )


process = process_from_source()  # Or process_from_so() / process_from_cfunc()

# Column c of the file is gene token c of the model's panel.
source = DataSource([(cpk_path, np.arange(len(gene_tokens), dtype=np.uint32))], len(gene_tokens))
schema = OutputSchema(
    np.arange(len(gene_tokens), dtype=np.uint32),
    dtype=np.float32,
    name="scgpt",
    process=process,
    # Row: [int64 gene_ids * L][float32 values * L][float32 target_values * L]
    row_bytes=length * 16,
)
loading = Loader(source, schema, workers=8, slot_rows=batch_size)

device = torch.device("cuda:0")
torch.cuda.set_device(device)
copy_stream = torch.cuda.Stream(device=device)

# Workers write directly into pinned memory, avoiding CPU field rearrangement
# and copies into separate pinned buffers.
layout = loading.ring_layout(slots=slots)
storage = torch.empty(layout.data_bytes, dtype=torch.uint8, pin_memory=True)
ring = Ring(
    layout,
    data=storage.numpy(),
)


def upload(raw):
    host = torch.from_numpy(raw)  # Shares ring memory; the array holds the slot lease
    try:
        with torch.cuda.stream(copy_stream):
            packed = host.to(device, non_blocking=True)
    finally:
        # Wait only for this copy stream so the GPU finishes reading the source
        # slot before this function returns.
        copy_stream.synchronize()
    packed.record_stream(torch.cuda.current_stream(device))
    return packed


# Import the model from the official scgpt package and use the sampling
# training configuration with GEP + MVC objectives.
model = TransformerModel(
    ntoken=len(vocab),
    d_model=512,
    nhead=8,
    d_hid=512,
    nlayers=12,
    vocab=vocab,
    dropout=0.2,
    pad_token="<pad>",
    pad_value=-2,
    do_mvc=True,
    explicit_zero_prob=False,
).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-4)
scaler = torch.amp.GradScaler("cuda")

with loading.start(accesses, ring=ring) as run:
    for batch in run:
        with batch:
            packed = upload(batch["scgpt"])
        # upload has dropped its source references and the batch lease has been
        # released, allowing workers to reuse the slot.

        # Split into views and make fields contiguous on the GPU for the model.
        gene_ids = packed[:, : length * 8].view(torch.int64).contiguous()
        values = packed[:, length * 8 : length * 12].view(torch.float32).contiguous()
        target_values = packed[:, length * 12 :].view(torch.float32).contiguous()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(
                gene_ids,
                values,
                src_key_padding_mask=gene_ids.eq(vocab["<pad>"]),
                MVC=True,
            )
            masked = values.eq(-1)
            loss = masked_mse_loss(output["mlm_output"], target_values, masked)
            loss += masked_mse_loss(output["mvc_output"], target_values, masked)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    run.join()
