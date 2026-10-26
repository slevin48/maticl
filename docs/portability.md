# Portability: one code base for MATLAB, Nelson and RunMat

maticl targets three runtimes with very different maturity. Nelson 2.0 runs
idiomatic MATLAB, but RunMat 0.6.2 has a strict static checker and several
semantic gaps. Instead of keeping one version per runtime, all of `src/` and
`tests/` is written in a subset that every runtime accepts. This page lists
that subset and why each rule exists, so contributors don't reintroduce
something that breaks one of them.

`runmat check <file>.m` runs the RunMat static checker without executing
anything. Run it on every changed file.

## Coding rules

| Rule | Instead of | Write | Reason |
|---|---|---|---|
| No `nargin` | `if nargin < 3, k = 4; end` | `function f(a, b, varargin)` + `if numel(varargin) > 0, k = cell2mat(varargin(1)); end` | RunMat: "local may be read before it is assigned" on any read of a possibly-missing input |
| No brace indexing of computed cells | `varargin{1}`, `s.blocks{b}`, `parts{2}` | `char(c(k))` for strings, `cell2mat(c(k))` for numbers, dynamic fields for structs | RunMat: "brace indexing requires a cell-like value" unless the cell is a literal in the same function |
| No cell / struct arrays of blocks | `M.icl.blocks{b}` | `M.icl.(sprintf('b%d', b))` | as above; `repmat` and `[s1, s2]` on structs are unsupported too |
| Initialise structs | `s.a = 1` on a new variable | `s = struct(); s.a = 1;` | RunMat: "Undefined variable" |
| Copy a field before vector or colon indexing | `pp.Xtr(:, f)` | `Xtr = pp.Xtr; Xtr(:, f)` | RunMat: "getfield: invalid index element" (scalar subscripts are fine) |
| No char indexing | `name(4:end)`, `key(1)` | `regexprep`, `strtok`, `strncmp`, or `char(codes(a:b))` with `codes = double(line)` | RunMat: "Slicing only supported on tensors" |
| Reshape linear indexing of column vectors | `v(3:end)'` | `reshape(v(3:end), 1, [])` | RunMat returns rows from linear indexing of column vectors, so a following `'` makes a column |
| Assign on every path | variable set only inside an `if`/loop and read later | assign a default first | RunMat definite-assignment check |
| No `switch` on strings | `switch method, case 'none'` | `if strcmp(method, 'none') ... elseif` | RunMat: "cannot convert CharArray to f64" |
| Concatenate with `cat` | `[A; B]`, `[a(:), b(:)]` on variables | `cat(1, A, B)` | RunMat: "cannot convert Tensor to f64" in some call contexts |
| No `persistent` | | | unsupported by RunMat |
| No capturing anonymous functions | `@(l) f(l, x)` | pass the data explicitly | RunMat compile error |
| No index assignment into logical arrays | `keep(j) = false` | numeric masks and `find` | RunMat: "Index assignment only for tensors" |
| Index with a row of indices from `find` | `X(:, keep)` | `X(:, keep(:)')` | RunMat's checker overflows its stack on some uses |
| Assign through a vector of indices from `unique` | `P(:, cls + 1) = p` | loop: `P(:, cls(i) + 1) = p(:, i)` | RunMat's checker overflows its stack |
| Avoid reserved words | `methods` as a variable | `methods_used` | RunMat parses it as a classdef keyword |
| Space around `.^` on literals | `2.^(0:2)'` | `2 .^ ((0:2)')` | RunMat reads `2.` as a number |
| `unique` on string arrays only | `unique(cellstr)` | `unique(string(c))` | RunMat |
| No `mfilename('fullpath')` inside functions | | `fileparts(which('tabicl_load'))` | RunMat returns the calling script's path |
| No `fseek` | | read sequentially | RunMat has no `fseek` |
| `fread(fid, n, 'single')` then `double(...)` | `'float32=>double'` | | Nelson returns wrong values for the `=>` form |

## Performance notes (RunMat 0.6.2)

* **Every statement costs time proportional to the data alive in the current
  function frame**, about 1 ns per element (≈ 27 ms per statement with the
  whole model in scope), and arguments are copied on each call. Only the
  current frame counts, not the caller's. This shapes the code:
  * `tabicl_load` reads one transformer block at a time in a helper function
    instead of slicing a single 27M-element buffer (134 s → 16 s);
  * each model stage receives only its own weights (`M.col`, `M.row`,
    `M.icl`), never the whole model;
  * loops in `tabicl_fit` live in helpers that do not see the model.
* RunMat compiles **every `.m` file in a path directory together**, and also
  the working directory, and it reports the first error against whatever
  function you happened to call. Run RunMat from a directory without stray
  `.m` files, and use `runmat check` on individual files to find the real
  culprit.

Nelson needs none of this, but nothing here costs anything there either.
