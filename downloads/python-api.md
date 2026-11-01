# Python API

The distribution and import package are `pycellpack`. The pure Rust crate is
`cellpack`; `crates/pycellpack` adapts its public interfaces through PyO3 and
NumPy into the private `pycellpack._cellpack` extension. Python composes
operations and chooses allocations; native code validates and executes them.
The binding crate's modules follow the core crate's (`store`, `format`,
`scans/`, `loader/`, `opt/`), and `_cellpack.pyi` declares the extension in
the same sections.

```text
src/pycellpack/
  _cellpack.pyi          declarations for the private compiled extension
  dataloader/            ScDataset, corpus Accesses and ordinary Tensor batches
    _dataset.py          lazy Store plus explicit standard dimension ids
    _accesses.py         Accesses dataclass and native cell-id conversion
    _loader.py           ScDataLoader iteration, cleanup and statistics
    _transfer.py         one packed CPU clone or pinned H2D and field views
  cellpack/
    _store.py            Store views, read/write/compress
    _file.py             File, Writer, prepared reads and read policies
    format.py            superblock, directory and chunk metadata
    selection.py         row normalization and GeneMap composition
    scans/               chunk/record codecs, tables, mapping and buffers
  loader/                sources, schemas, accesses, rings and runs
    access_fn/           AccessFn, C ABI and Numba signatures
      include/           access_fn_cabi.h
    opt/                 plans, options, cost models and typed summaries
    costs/               cost records, resolution and calibration
    process_fn/          ProcessFn, compile_library, numba bindings
      include/           process_fn_cabi.h and template/<model>.hpp presets
  io/
    _cpk.py              .cpk tools: open/read/iter/write, cpk_info
    _anndata.py          H5AD/H5CP read and write, h5ad_to_cpk (optional deps)
  anndata/               optional AnnData subclass (class definitions only)
```

The root exports `CSR`, `File`, `Store`, `Writer`, `read`, `write`, `compress`,
`DataSource`, `Accesses`, `Loader`, `LoaderConfig`, `OutputSchema`, `ProcessFn`,
`DistributedLoader`, `load`, `compile_library`, `Plan`, `PlanOptions`,
`read_h5`, `write_h5ad`, `write_h5cp`, `open_matrix`, `ScDataset`, `ScDataLoader`
and `AccessFn`, plus the `cellpack`, `dataloader`, `loader` and `io`
modules (plans and costs are `loader.opt` and `loader.costs`). Codecs are under `cellpack.scans`.
Importing the root loads no SciPy, AnnData, h5py, pandas, PyTorch or Numba.
Those integrations import their dependencies when used.
The root `Accesses` is the corpus dataclass. The native global-cell sequence
is available explicitly as `loader.Accesses`.

## Training DataLoader

`pycellpack.dataloader` provides `ScDataset`, an `Accesses` dataclass,
`AccessFn`, and `ScDataLoader`. It uses the native loader directly, with
native worker threads. Torch is imported when iteration starts, not by
import, construction, `repr`, or `len`. Install Torch separately for the
target environment; it is not a distribution dependency or extra.

```python
from pycellpack.dataloader import Accesses, ScDataLoader, ScDataset

corpus = [
    ScDataset(adata_a.X, ids=ids_a),
    ScDataset(adata_b.layers["counts"], ids=ids_b),
]
loading = ScDataLoader(
    corpus=corpus,
    accesses=Accesses(dataset_ids, rows),
    out_ids=panel_ids,
    process=process,
    batch_size=256,
    device="cuda:0",
    prefetch=4,
    cache_bytes=512 << 20,
    contiguous=("genes", "values"),
)
with loading:
    for batch in loading:
        train_step(batch["genes"], batch["values"], batch["targets"])
```

ScDataset accepts a file-backed 2-D Store and a NumPy integer vector `ids`
covering its **visible** columns. IDs describe the caller's standard dim
space; -1 or uint32.max drops a dim, and other ids must be distinct.
Ordinary row and column subsets/reorderings remain lazy. Rows in access
sequences address this dataset view, not original file row numbers. Repeated
mapped columns, dtype-converting Store views, and in-memory streams are
rejected explicitly; write a memory stream to `.cpk` before constructing a
dataset. Choose the mapped dtype through the loader's `dtype`, rather than
`Store.astype`.

`ScDataset.from_anndata(adata, ids=..., layer=None)` selects X; a layer name
selects `adata.layers[layer]`. It imports no AnnData and encodes no matrices
implicitly. H5CP `read_h5(..., lazy=True)` preserves CellPack matrices as
Stores, including selected views. Plain H5AD matrices require explicit
conversion to CellPack.

corpus order fixes dataset IDs. The loader keeps a tuple snapshot of its
members and snapshots each dataset's id map. `out_ids` defines the common
panel, in ProcessFn table order, and can be omitted when all datasets have
the same id map. Sparse standard ids are lowered to requested panel positions,
so reservations do not grow with their largest external value. dtype defaults
to the process template's required dtype, or float32. Without `process`,
batches contain dense `X`. Processes need a declared OutputLayout; custom
native callbacks can request worker scratch with `scratch_bytes`.

### Access inputs

`Accesses` is a dataclass with two fields: `dataset_ids` and `rows`. Both are
1-D NumPy integer arrays of the same length. The caller owns the arrays and
all sampling, ordering, shuffle, epoch and evaluation policy. `Accesses`
holds their references and `len(accesses)` reports the sample count.

Provide exactly one of:

- `accesses=Accesses(dataset_ids, rows)`, or the tuple `(dataset_ids, rows)`.
- `access_fn=AccessFn(...)`, a native range function bound through the C ABI.

There is no default traversal or internal sampling factory. Fixed arrays
are validated and snapshotted when each iterator starts. Edits to the caller's
arrays affect subsequent iterations while the current native run retains
its snapshot. The caller can replace `loading.accesses` or `loading.access_fn`
between iterations; keep exactly one configured. For example, each training
epoch can prepare a new order externally and assign it to `loading.accesses`.

`len(loading)` uses the current input length without generating accesses.
`drop_last=True` trims to a batch multiple, using a native interval view for
AccessFn. The current core's CellId remains uint32; corpus addressing does
not widen registered cell IDs. Access counts can exceed cell counts through
caller-defined repetition.

### Batch ownership and cleanup

CUDA output uses a pinned native ring and one complete packed H2D per batch.
Only its dedicated copy stream is waited on before releasing the native Batch
and dropping NumPy/host Tensor views. Returned field views retain device
storage only. `contiguous=False` returns field views; True or field names
request copies on independent field streams, with dependencies and allocation
lifetimes recorded for the current consumer stream. Unaligned layouts use a
field byte copy after the packed transfer. The CUDA path performs field
packing and contiguous copies on the GPU.

CPU output uses one owned packed clone before field views, so saved batches
cannot hold native ring slots. Explicit contiguous fields may add copies.
The default device is CPU. `prefetch` counts bounded host slots; `workers`
counts native threads.

One iterator is active at a time. Exhaustion joins the run, saves its RunStats
as `loading.stats`, and closes it. Exceptions, iterator/loader `close`, and
context exit cancel unfinished native work and join cleanup. `with loading:`
closes early breaks; `with iter(loading) as it:` provides iterator-scoped
cleanup. Saved output Tensors remain valid. The loader can be iterated again.
Pickling requires the active iterator to be closed and serializes AccessFn's
construction recipe rather than a bound native address.

See [the runnable DataLoader example](../example/dataloader-example.py).

## Native access functions

`loader.access_fn.AccessFn` owns and binds functions from a shared library,
C/C++ source or Numba (`from_library`, `from_source`, `from_cfunc`), following
ProcessFn's construction and ownership pattern. `include_dir()` exposes
`access_fn_cabi.h`; the matching core header is
`crates/cellpack/include/access_fn_cabi.h`.

```c
typedef int32_t (*CellPackAccessFn)(void *context, uint64_t start,
                                  uint64_t count, uint32_t *cells);
typedef int32_t (*CellPackCorpusAccessFn)(void *context, uint64_t start,
                                        uint64_t count, uint32_t *dataset_ids,
                                        uint64_t *rows);
```

Bind the first with `kind="global"` (default) and the second with
`kind="corpus"`, giving `length` separately. Calls fill exactly count entries
for `[start, start+count)`, are deterministic and thread-safe, and may repeat
or arrive out of order from workers and the online compiler. No borrowed
output pointer may escape and no exception may cross the C boundary. A
nonzero status fails the run. Corpus ids are translated against dataset view
row counts. Per-thread adapter row scratch is outside MemoryEstimate's known
reservations. Sampling policy and any seed/epoch configuration are external.

```python
from pycellpack.loader.access_fn import AccessFn

access_fn = AccessFn.from_source(
    "sampling.cpp", function="generate_accesses", kind="corpus",
    length=n_samples, user_data=tables, key=table_values, owner=table_arrays,
)
loading = ScDataLoader(corpus=corpus, access_fn=access_fn, ...)
```

The include path and content cache include the access header and explicitly
supplied dependencies. The library, ctypes/NumPy user_data and additional
owner are retained. key or explicit identity covers configuration values;
pointer addresses are not stable configuration identities. Picklable recipes
reconstruct functions after pickle; process-local owners need external
reconstruction. `numba_bindings()` lazily provides GLOBAL_SIGNATURE and
CORPUS_SIGNATURE for the same pointer ABI.

ScDataLoader binds corpus row counts automatically. Lower-level Loader
start/compile/replay and PageTrace also bind AccessFn to their source's
matrix/view counts. `bind(counts)` returns native Accesses explicitly;
`fill(..., counts=...)` inspects intervals as global cell ids.

The raw-address `loader.Accesses.function` remains available.
`loader.Accesses.slice(start, count)` composes native intervals without
materializing an access function.

## Stores and streams

`Store` is the array API. `File` is the low-level stream handle: a `.cpk`
file, a bounded embedded region, or an in-memory stream from `File.from_bytes`.
`write(path, matrix, ...)` and `compress(matrix, ...)` return Stores; the
latter keeps an anonymous in-memory stream, accessible as `store.file`.

```python
import numpy as np
from pycellpack import cellpack

x = np.arange(24, dtype=np.float32).reshape(6, 4)
store = cellpack.write("matrix.cpk", x, chunk_rows=3)
view = store[[5, 1, 5], [3, 0, 3]].astype(np.float64)
out = view.read_dense()                  # shape (3, 3); reads on demand
csr = view.read_csr()                    # NumPy-backed cellpack.CSR
selected = cellpack.read("matrix.cpk", [5, 1], [3, 0], format="dense")
encoded = store.file.to_bytes()          # the complete .cpk stream
memory = cellpack.Store(cellpack.File.from_bytes(encoded))
```

Store indexing is orthogonal and keeps both axes, including integer selectors.
Both axes take slices (also negative steps), integer arrays with repeats and
boolean masks. Indexing, `astype` and `map_genes` compose views without data
I/O. Streams keep no gene or cell names: positions and alignment are the
caller's. Higher-rank dense arrays use `Store(file, shape=...)`, with trailing
axes flattened in the stream; these views select cells only.

`read(format=None)` defaults to the stored layout and returns a NumPy array
or `CSR`; `read_dense` and `read_csr` choose it explicitly. `to_memory`,
`toarray`, `tocsr` and `np.asarray(store)` provide NumPy/SciPy conveniences.
`iter_batches(batch_size=4096, format=None, ...)` decodes successive cell
batches. `store.file`, `cells`, `genes`, `shape`, `dtype`, `format` and
`is_identity` describe the view; selection arrays are read-only.

`map_genes(gene_map, n_genes=None)` maps this view's columns into a panel:
integer targets must be distinct; `-1` or `scans.UNMAPPED` drops a column.
Missing panel columns read as zero. `cellpack.GeneMap` owns a validated map;
`cellpack.selection.as_gene_map` accepts it, an integer vector or a dict
from source positions to targets. `normalize_rows(rows, size)` returns
uint64 positional IDs, preserving order and repeats.

### File tools in `io`

```python
from pycellpack import io

store = io.write_cpk("matrix.cpk", x)                # matrix, Store or row batches
io.write_cpk("copy.cpk", (x[i:i + 2] for i in range(0, 6, 2)))
info = io.cpk_info("matrix.cpk")                     # shape, dtype, format, chunks, nbytes
part = io.read_cpk("matrix.cpk", [5, 1], format="csr")
for batch in io.iter_cpk("matrix.cpk", batch_size=2):
    ...
lazy = io.open_cpk("matrix.cpk")                     # a Store; also File, bytes
```

`write_cpk` streams a Store or an iterable of row batches through `Writer`
without holding more than one batch; the stored layout defaults to the
input's. `cpk_info` reads the superblock only. `io.h5ad_to_cpk` (AnnData
extra) exports one matrix slot of an H5AD/H5CP file the same way.

### Prepared reads and caller buffers

```python
file = cellpack.File("matrix.cpk")
mapping = cellpack.GeneMap([1, -1, -1, 0], 3)
selection = file.prepare_read([5, 1, 5], gene_map=mapping, dtype=np.float64)
workspace = cellpack.ReadWorkspace()
out = np.empty(selection.shape, selection.dtype)
selection.read_dense(out=out, workspace=workspace)
csr = selection.read_csr(index_dtype=np.int32, workspace=workspace)
workspace.clear()

# A Store composes the same read, including its selections and dtype.
prepared = store[[5, 1, 5], [3, 0]].astype(np.float64).prepare()
prepared.read_dense()
```

`ReadSelection.capacity_nnz` is the source CSR capacity, not the exact number
of mapped entries; dropped genes can reduce the returned `CSR.nnz`.
`ReadSelection.shape`, `dtype` and `io_summary` describe it. Dense `out=`
returns the supplied array; CSR `out=` takes a `CSR` or `(indptr, indices,
data)` tuple with sufficient capacity. Index arrays default to int64;
int32, uint32 and uint64 are also accepted (native CSR reads additionally
accept uint16 column indices). Stored sparse zeros remain stored entries.
A Store with repeated genes expands the decoded unique columns on read;
it cannot `prepare()`, or read repeated genes into caller CSR buffers.

A workspace retains scratch and permits one active read; concurrent reads
need separate workspaces. Numeric buffers must be native-endian, aligned,
C-contiguous and writable for output. Overlapping input/output/scratch is
rejected. Do not mutate or resize borrowed arrays during a native call;
I/O or codec failures can leave partial output, which must be discarded.
Storage codecs preserve the stored dtype unless a view/selection requests
conversion. Encoding/decoding uses `workers` threads, defaulting to one per
available CPU, at most 32 (`PYCELLPACK_WORKERS` overrides).

`File.prepare_read(..., read_options=ReadOptions(...))` controls payload
reads: `strategy="ranges"` coalesces selected byte ranges, `"whole"` reads
whole record regions and `"auto"` compares bytes plus an explicit per-request
cost. `max_gap_bytes=0` and `request_cost_bytes=4096` are defaults, not a
calibrated device model. `ReadSummary` reports `record_bytes`, `read_bytes`,
`requests` and `whole_chunk_bytes`, excluding metadata preparation.
`File.chunks`, `read_chunk(chunk, out=None, metadata_only=False)` and
`chunk_meta(chunk)` expose stream components; `offset` and `length` bound
embedded regions.

### Writing and file identity

`Writer(path, n_cols, dtype, format="csr", chunk_rows=4096, page_size=4096,
options=None, workers=None, atomic=False)` streams `append(matrix)` calls,
then `finish()`; successful context exit finishes and exceptional exit aborts.
Input CSR must be canonical with sorted distinct indices. `write` and
`compress` also accept ordinary NumPy/SciPy matrices and canonicalize sparse
input. `options` is a `scans.EncodeOptions`.

`atomic=True` writes a sibling temporary file and replaces the destination
only after success. Aborting or dropping the writer discards that temporary;
existing permission bits are preserved, new atomic files use mode 0600.
Writers belong to their creating process. Atomic replacement requires a
regular path and does not guarantee persistence across power loss.

An opened File continues reading its original descriptor after path
replacement. `to_bytes()` and unpickling reject a replaced path; in-memory
Files pickle by their bytes. Stores pickle their file and composed view.

## Chunk and record codecs

```python
from pycellpack.cellpack import scans

chunk = scans.encode_chunk(x, options=scans.EncodeOptions())
meta = scans.Meta(chunk, source_dtype=x.dtype)
tables = scans.DecodeTables(meta)
workspace = np.empty(meta.decode_workspace_words, np.uint64)
dense = scans.decode_dense(meta, tables=tables, workspace=workspace)
csr = scans.decode_csr(meta, tables=tables)
row = scans.decode_record(meta, cell=2)
parts = scans.decode_parts(meta, cell=2)
panel = scans.map_dense(parts, [1, -1, -1, 0], n_genes=3)
```

Raw chunks record value width, not signedness or floating interpretation;
pass `source_dtype=` to `Meta` and to `decode_chunk(raw_bytes, ...)`.
`File.chunk_meta` supplies the stream dtype. `decode_chunk(meta, ...)` can
reuse parsed metadata; `dtype=` selects the output dtype.

| Surface | Objects and operations |
|---|---|
| Metadata | `Meta`: shape, sizes, records, cell layout/nnz, column modes |
| Tables | `EncodeTables(meta)`, `DecodeTables(meta)`, `compose_mapping` |
| Encoding | `chunk_sizes`, `encode_chunk`, `encode_record`, `record_bound`, `encode_workspace_words` |
| Decoding | `decode_chunk`, `decode_dense`, `decode_csr`, `decode_record`, `decode_record_dense`, `selection_nnz`, `dense_bytes`, `decode_workspace_bound` |
| Parts and mapping | `decode_parts`, `Parts`, `Dict`, `ModeDict`, `Ranks`, `Values`, `cast_values`, `map_parts`, `map_parts_at`, `map_dense` |
| Paged input | `PagedRecord`; record/parts decoders accept it as their record |
| Packing/dispatch | `packed_len`, `pack_chunk`, `record_nnz`, `kernel_set`, `supported_kernel_sets` |

Codecs accept caller `out=` and `workspace=` buffers. Workspace words are
uint64. `chunk_sizes` returns `ChunkSizes(output_bytes, workspace_words)`;
`encode_chunk` queries only missing buffers. `fill="assume_zeroed"` skips
clearing a supplied dense output, so the caller must provide zeros.

`scans.PreparedEncoder(matrix, rows=None, options=...)` snapshots its inputs,
reuses the encoding model and exposes `output_bytes`, `workspace_words` and
`meta`; `encode(out=..., workspace=...)` writes a chunk. Its pre-encoding
metadata is suitable for encoding tables, not cell-indexed decoding.
Rust's PreparedEncoder instead borrows immutable input.

`scans.Decoder(meta, gene_map=..., n_genes=..., dtype=...)` binds dtypes,
composes mapping once and reuses scratch for `decode_dense`, `decode_csr`
and `decode_record`. Use separate decoders for concurrent calls. Owned
`Parts` keep immutable private metadata, record/page copies and scratch;
their component arrays are read-only snapshots. They are not borrowed
zero-copy Rust Parts. Explicit `dtype=` must agree with supplied output.

## Sources, schemas and loaders

A DataSource registers `(source, ids)` pairs and a positive `vocab`.
`ids[c]` is the standard id of stored column c, below vocab; `-1` or
`0xFFFFFFFF` drops it. The caller supplies every mapping. Sources are paths,
File regions or Stores selecting cells only (gene views are rejected).
Global cell IDs number contributed cells in registration order; selected
Stores are translated to stored IDs natively for runs, plans and traces.
In-memory streams are not loader sources. `add(source, ids)` returns the
new matrix's first global cell ID; existing loaders retain their snapshot.

```python
from pycellpack import loader

source = loader.DataSource([("matrix.cpk", np.arange(4))], 4)
outputs = {"pair": loader.OutputSchema([3, 0], np.float64)}
loading = loader.Loader(source, outputs, workers=2, backend="pread",
                        inflight=8, slot_rows=2)
with loading.start([5, 1, 5, 0], slots=2) as run:
    rows = [batch["pair"].copy() for batch in run]
    stats = run.join()
```

`Loader(source, outputs=None, config=None, **settings)` takes one schema,
a sequence or a name-to-schema mapping; the default is float32 over every
standard id, named X. `schemas` is a read-only name-to-schema mapping and
`names` preserves output order. Explicit schema ids remain; unset ids take
the smallest unused values. Names and ids must be distinct.

`OutputSchema(out_ids, dtype=np.float32, id=None, name=None, default=0,
process=None, row_bytes=None, out_dtype=None, scratch_bytes=0)` describes
one output per position. Built-in rows are dense; existing source columns
missing a CSR entry give zero, while genes absent from the source take the
exact typed `default`. `allowed(source_dtype, target_dtype)` queries the
loader conversion contract. Process outputs occupy fixed `row_bytes`:
`batch[name]` is a uint8 matrix unless `out_dtype` gives a numeric view;
`batch.region(name)` always returns bytes.

`LoaderConfig` fields are `workers`, `backend`, `inflight`, `claim_batch`,
`slot_rows`, `window`, `profile` and `cpus`. Pass a config or its fields,
not both. Workers default to usable CPUs, at most 16; backend probes
io_uring and falls back to pread. `DataSource(..., direct_io=None)` probes
O_DIRECT support, falling back to buffered I/O; `direct_io=False` forces it.
The page size defaults to the first stream's (512 B to 64 KiB).
`loader.load(source, cells=None, out=None, config=None, **settings)` is the
one-shot float32 full-panel convenience.

### Accesses and plans

One access is one schema at one position: `p0s0 p0s1 ... p1s0 ...`.
Arrays are one-dimensional integer cell IDs in uint32 range; order and
repeats remain. `loader.Accesses` supports stored arrays, computed shuffle/subset
sequences, native functions, `replicate(sequence, schemas)` and
`offset(sequence, offsets)`. Function addresses retain an explicit `owner`
when their lifetime needs it. `fill(start=0, count=None, out=None)`
materializes an interval. For multiple outputs, replicate or offset each
position's cell rather than passing just one cell per position.

| Loader operation | Result |
|---|---|
| `plan_options(cache_bytes, cost="auto")` | `PlanOptions` using this loader's gap and resolved costs |
| `compile(accesses, options, online=False)` | offline Plan; `online=True` simulates finite-horizon compilation and attaches `online_stats` |
| `compile_stats(accesses, options)` | `OnlineStats` without retaining a plan |
| `replay(plan, accesses, cost=None)` | `PlanSummary`, optionally repriced without changing the plan |
| `start(accesses, plan=None, ring=None, slots=None, world=None, rank=0)` | asynchronous Run |
| `memory_estimate(plan=None, slots=4, world=1)` | `MemoryEstimate` of known reservations |

`options` accepts PlanOptions or a byte budget. `start(plan=...)` accepts an
offline Plan, PlanOptions or a byte budget for compilation during execution;
None means no cache. Page and native-precision row caches share capacity;
outputs are never cached. `PlanOptions` contains `cache_bytes`, `gap`,
`threads`, `objective`, `cost`, `rows`, `block_pages`, `row_policy`,
`horizon`, `chunk` and `cpus`. Rows and block reads require a cost model;
`row_policy="global"` is an offline reference, while `"incremental"` is
the default. Gap, horizon and window count slots; chunk counts accesses.
Compiler threads default to usable CPUs offline, a quarter of workers
online, at most 16.

Plans bind source mapping, schemas, slot layout and access sequence. Plan
supports `open`, `write`, `to_bytes`, `from_bytes`, pickling, `summary`,
`online_stats`, `schemas`, `slot_rows`, `distance` (accesses), `reuse_gap`
(slots), `cache_bytes`, `cache_frames` and `residencies()` arrays.
`loader.opt.PageTrace.from_pages` / `from_source` and `optimum(capacity_pages)`
provide the serial Belady reference, not a claim of optimal row planning.

Memory estimates include cache/generation, private frames, worker scratch,
PlanRing, known metadata and output ring. Compiler state, allocator overhead
and OS cache are excluded; the estimate is not RSS or a complete memory
limit. Starting warns when known reservations exceed available memory.

### Rings, leases and lifecycle

`loading.ring_layout(slots=4, world=1)` returns a picklable RingLayout;
`loading.ring(...)` allocates it. `Ring(layout, data=None, control=None,
initialize=True)` also accepts caller-owned or shared uint8 arrays.
Data requires 8-byte alignment, control 64-byte alignment. The layout
reports `control_bytes`, `data_bytes`, `data_min_bytes`, `offsets`,
`slot_stride`, `row_bytes`, `slot_rows`, names and dtypes. A ring supplied
to start must match the loader's output names, native dtypes and row sizes.

Batch b fills slot `b % slots` and belongs to rank `b % world`. Every schema
shares one ring. Batch offers mapping access, `arrays()`, raw `region`,
`index`, `rows`, `first_position`, `generation`, `held`, `release()` and
`observer()`; `.data` is the single-output convenience. Advancing iteration
releases its previous Batch; a `with batch` block releases on exit.
Exported NumPy views, slices and tensors retain the slot lease until dropped.
Retaining a view applies backpressure; copy it to keep data independently.
Device transfers must retain the host view until the device finishes reading.

Run is its rank's iterator; `run.consumer(rank)` opens another rank's
Consumer. `run.wait(batch)` and `consumer.wait(batch)` acquire individual
batches, returning None past the end; release those batches explicitly or
drop them. Consumer iteration releases its own preceding Batch. Closing a
Consumer records whether it consumed its rank completely. Participant
records live in the ring control storage, without another consumer array.

`Run.join()` returns typed RunStats. Consume slots before joining unless
they all fit or another thread/process drains them; otherwise join detects
an impossible local stall and cancels with a reason. `cancel()` requests
cancellation; `close()` cancels unfinished work, releases the iteration-held
Batch and waits for native cleanup without holding the GIL. Context exit
closes. Exported views remain valid. `finished` includes success, failure
and cancellation; `succeeded` distinguishes success, also after close.
`wait_finished(timeout_ms=50)` supports 0–1000 ms and Python signals.
`join()` remains available after close; active operations such as snapshot
require an open run. Rings can serve successive runs once leases are gone;
`ring.abort()` is permanent.

`ValueError` reports invalid options, sequences or plans, `OSError` reports
I/O failures and `RuntimeError` reports worker/process failures, busy loaders
or cancellation. Async failures retain those categories.

### Profiling and costs

Every wheel carries untimed and timed worker/planner instances. `profile=False`
disables timing, True selects all sites, and a site/group name or a sequence
selects a mask. `profile_sites()`, `profile_groups()` and `build_info()`
describe them. Event counts are always kept. `run.snapshot()` and `join()`
return RunStats with typed `io`, named `outputs`, optional `profile` and
`online`. Records support dict conversion for JSON and interprocess use.
ProfileReport exposes `site`, `time`, `map_time` and `process_bands`; sites
report inclusive/self times, details and optional histograms.

CostModel, InputCost, RowCost, OutputCost and ProcessCost are typed values.
The model prices device requests, device bytes, worker cores and row-memory
bandwidth; output costs are keyed by loader output name or position. See
[cost model and calibration](architecture/cost-model-and-calibration/README.md)
for record resolution, calibration and CLI. No calibration is implied by
default costs.

Stored cost records contribute their known fields; missing host fields use
the current defaults, and unrelated fields are ignored. Reading a partial
record does not rewrite or recalibrate it.

### Process functions

ProcessFn wraps the `pycellpack.process.v1` capsule ABI in
`loader/process_fn/include/process_fn_cabi.h` (also `ProcessFn.include_dir()`).
`from_library`, `from_source` and `from_cfunc` bind exported C functions,
compiled C/C++ source or Numba callbacks. `compile_library` compiles and
caches libraries by source, compiler, options, header and dependencies.
`row_format="auto"` requires CSR and dense callbacks with equivalent
results; `"csr"` or `"dense"` requires only that format. `partial_rows=True`
enables the CSR resolver. Nonzero callbacks fail the run.

Callbacks receive mapped columns and values converted to the schema dtype;
they write only their row_bytes and scratch, retain no native pointers and
must be thread-safe. `identity` versions the function and all configuration
it reads; `key` contributes configuration to derived identities. `owner`
keeps native buffers alive. Loader/schema/process objects pickle when their
sources and factory recipes do; raw pointers or capsules cannot be sent to
another process. Numba is optional and loaded on use
(`loader.process_fn.numba_bindings()`). See
[the illustrative model example](../example/loader-example.py).

### Named process outputs

`process_fn.OutputLayout` describes a fixed output record using NumPy
structured dtype fields. It uses C struct alignment and tail padding by
default; `align=False` describes packed records. Fields must contain
native-endian numeric values. Explicit dtype offsets and itemsize are also
supported. The declaration must match the native function's actual writes.

```python
from pycellpack.loader.process_fn import OutputLayout, ProcessFn

layout = OutputLayout([("ids", "i8", (1024,)), ("length", "u4")])
# Add output=layout to from_library, from_source or from_cfunc.
process = ProcessFn.from_library("tokenize.so", dense="tokenize_dense", output=layout)
schema = process.schema(out_ids, name="tokens")
```

`process.output` holds the layout; `row_bytes` is its padded record size.
`process.schema(...)` builds an OutputSchema with that size and defaults
to float32 mapped input for custom functions. `OutputSchema(...,
process=process)` also infers row_bytes when a layout is declared and rejects
an explicitly conflicting size. Opaque functions without a layout keep the
explicit row_bytes interface.

`process.unpack(batch.region(name))` returns a dict of NumPy field views.
The input must be `uint8[..., row_bytes]` with a contiguous byte axis; leading
axes may be strided. Scalar fields have the leading shape, array fields
append their declared shape. Views keep the source and its slot lease alive,
including after the Batch is released or the Run closes. Fields may be
strided or unaligned. `copy=True` returns independent, aligned C-contiguous
arrays when a caller needs to retain CPU data. This helper accepts NumPy host
arrays.

For GPU training, retain the single packed NumPy slot view and construct a
zero-copy host Tensor with `torch.from_numpy`. Transfer the complete packed
buffer once, then construct field views on the GPU and make only the needed
fields contiguous there, using separate CUDA streams when useful. Allocate
the output ring over pinned memory to support asynchronous H2D. Wait for
the H2D copy's event or copy stream to complete before releasing the Batch
and dropping its host Tensor and NumPy view; submitting a nonblocking copy
alone does not finish reading the slot. Every additional CPU field view
also retains the slot until dropped. Device fields retain device storage
only, so their processing can continue after the host slot is reused.
The [runnable example](../example/processfn-example.py) demonstrates this
path with `--device cuda`, including stream dependencies and allocation
lifetimes. CPU inspection uses zero-copy `unpack()` views.

### Python model presets

`loader.process_fn.presets` binds the four supplied C++ templates directly:

```python
from pycellpack.loader.process_fn import presets

process = presets.scgpt(tokens, pad_token=0, cls_token=1, length=1024, seed=7)
schema = process.schema(out_ids, name="scgpt")
loading = Loader(source, schema, workers=4, slot_rows=128)
with loading.start(cells) as run:
    for batch in run:
        with batch:
            inspect_fields(process.unpack(batch.region("scgpt")))
    run.join()
```

This CPU inspection callback must finish using its field views before
returning. Use the packed-transfer path above for GPU training.

Tokens and auxiliary tables are indexed by **schema output dim**, in the
same order as `out_ids`; negative tokens exclude a dim. The caller still
owns source gene alignment. Factories copy their inputs and retain all
native owners. Their pickle recipes store values, rebuild pointers in the
receiver and reuse the compilation cache. Identities cover the code,
template parameters and table contents, independently of pointer addresses.
`row_format="auto"`, `"csr"` and `"dense"` choose input delivery as for other
ProcessFns. Compiler keywords go to `compile_library`; a C++17 compiler is
required even when resolving a cached build. Native and Python struct sizes
and offsets are checked during compilation.

| Factory | Required tables and token configuration | Output fields |
|---|---|---|
| `scgpt` | `tokens`, `pad_token`, `cls_token` | `genes`, `values`, `targets` |
| `geneformer` | `tokens`, `medians`, `pad_token`, `cls_token`, `eos_token` | `ids`, `length` |
| `uce` | `tokens`, `chroms`, `starts`, `pad_token`, `cls_token`, `chrom_offset`, `chrom_close_token` | `ids`, `loss_ids`, `labels`, `length` |
| `state_se` | `tokens`, `global_size` | `ids`, `tasks`, `targets`, `counts`, `total` |

The defaults and sampling behavior are those of the shipped templates,
not inferred from a model checkpoint. Each factory exposes its template's
hyperparameters as keywords. `input_dtype` is float32 and `out_dim` is the
table width; schemas reject incompatible mapped dtypes or widths before
loading. STATE SE's `batch` controls shared query draws over consecutive
access positions and must match the intended training microbatch. Cell
content restrictions and native status codes still follow each header.
See the [runnable CPU/CUDA example](../example/processfn-example.py).

`include/template/<model>.hpp` holds model presets (`scgpt`, `geneformer`,
`uce`, `state_se`): C++ templates over a parameter struct, because each
training configuration needs its own statically compiled functions. A source
derives the parameters from the preset's defaults and exports an instance:

```cpp
#include "template/scgpt.hpp"
struct Config : pycellpack::preset::ScGptDefaults { static constexpr uint32_t length = 1200; };
PYCELLPACK_EXPORT_PROCESS_FN(scgpt, pycellpack::preset::ScGpt<Config>)
```

`ProcessFn.from_source(code=..., csr="scgpt_csr", dense="scgpt_dense",
user_data=tables, key=...)` binds it; each header documents its runtime
tables (the `user_data` layout), output layout (`out_bytes` is the schema's
`row_bytes`) and status codes. `template/common.hpp` adapts presets to the
C ABI.

## Shared and distributed loading

`SharedRing.create(layout, transport="auto", extra=None)` owns shared control,
data and optional named byte regions; extra names cannot replace control or
data. `descriptor` is picklable, `attach(descriptor)` maps it in another
process, `inheritable()` prepares child memfd descriptors and `consumer(rank)`
opens a rank's Consumer. Close removes owned names and drops descriptors;
exported arrays retain their mappings.

`DistributedLoader(loader=None, transport="auto")` composes any picklable
Loader, including named/process outputs, over a SharedRing. On group rank
zero, give a Loader or a factory returning one; other ranks give nothing.
`start(accesses=None, plan=None, slots_per_rank=2, drop_last=False, group=None)`
is collective over an initialized torch.distributed group on one host.
Rank zero supplies the sequence and starts a dedicated CPU producer; every
rank consumes its batches. A plan may also be a plan-file path. Array-only
`drop_last` trims to complete rounds of all ranks' slots.

DistributedRun provides iteration, `wait`, `join`, `cancel`, `close`,
`finished`, rank/world, length and producer_pid. Early context exit or a rank
failure cancels every rank. Producer failures propagate through a locked
status region and rank-zero process supervision. The status region is sized
for output names and profile details; an oversized report is an explicit
producer error, never a successful run without statistics.

Transport auto tries `/dev/shm`, anonymous memfd and then mapped files under
`PYCELLPACK_SHM_DIR`. `transport=` or `PYCELLPACK_SHM` forces one. memfd
peers receive descriptors over a same-user Unix socket; inherited descriptors
use POSIX record locks for status synchronization. The producer initializes
neither CUDA nor the training process group. Torch is optional.

## AnnData and H5CP

`pycellpack.anndata.AnnData` subclasses the pinned AnnData 0.13.4 (Python
>=3.12). X, layers, obsm, varm, obsp, varp and raw accept immutable Stores or
Files; views compose selections and `to_memory()` decodes them. Importing
this module registers the CellPack encoding with AnnData; files are read and
written through `pycellpack.io`.

`io.read_h5` reads plain H5AD or H5CP; `io.h5ad_to_cpk(path, out, key)`
exports one matrix slot to a `.cpk` file in row batches; `lazy=True` or `backed="r"` returns
Stores for embedded streams. `open_matrix(path, key)` opens an embedded
File. `write_h5cp` embeds Stores by default, encodes ordinary numeric matrices
with `encode=True`, or decodes with `encode=False`. `write_h5ad` always writes
plain H5AD, decoding Stores. Whole Stores copy byte-for-byte; sliced ones
re-encode. Streams are contiguous unfiltered uint8 HDF5 datasets aligned for
native page I/O; compression arguments apply to other AnnData datasets.
H5CP writes stage beside the destination and replace on success, including
when the lazy source reads the destination being replaced.
