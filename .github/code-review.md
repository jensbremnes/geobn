# Code review guidelines for geobn

geobn is a Python library for Bayesian network inference over geospatial data. Data
sources (rasters, remote APIs, scalars) are aligned to a grid, discretized, and used as
evidence for pixel-wise inference, producing posterior probability and entropy rasters.

The library is a thin, domain-agnostic layer between geodata and an existing Bayesian
network: it does the alignment, discretization, and batching, and leaves the model and
what happens to the output to the caller. Sources are declarative, so constructing one
performs no I/O and data is fetched when inference runs. Results come back as numpy
arrays, xarray Datasets, or GeoTIFFs rather than bespoke types. Weigh findings against
that: a change that widens the library's remit, adds a type the caller has to learn, or
moves work behind a layer the caller cannot reach runs against it.

The library is pre-1.0 and has no users. Source lives in `src/geobn/`, tests in `tests/`,
prose docs in `docs/` and `README.md`. Formatting and lint are handled by ruff
(line length 100, rules `E`, `F`, `I`). Tests run under pytest.

Post findings that need action. Rank them by how much they would cost if left in.

## What to look for

**Correctness.** Logic errors, wrong numerics or units, CRS and coordinate-order
mistakes, transforms applied to the wrong space, off-by-one in grid indexing, silent
failure modes, and cache and TTL edge cases.

**Hallucinated APIs.** Calls to functions, methods, or keyword arguments that do not
exist in the library or its dependencies (numpy, rasterio, pyproj, xarray, pgmpy,
affine, folium, matplotlib, requests), or that are used with the wrong signature or
return type. Check the call against the definition rather than against what the name
suggests.

**Hollow tests.** Tests that assert on mocks instead of behaviour, that cannot fail, or
that cover only the happy path. Numeric assertions with missing or absurdly loose
tolerances.

**Over-abstraction.** New classes, wrappers, protocols, or config options with only one
caller or one implementation. Parameters that no caller sets to anything but the default.

**Duplication.** New helpers that reimplement something already in the repository, such
as a second distance function or coordinate transform. Name the existing function.

**Defensive noise and dead code.** Checks on values that cannot be wrong, broad
`try`/`except` blocks, and fallbacks that hide failures. Unused imports, parameters, and
functions, and comments that restate the code.

**Inconsistency with the library.** New code that handles errors, naming, CRS, or return
types differently from the rest of the library. Data sources should follow the
`DataSource` ABC in `src/geobn/sources/_base.py` and the patterns of the sources beside
it.

**Scope.** geobn is a geodata and Bayesian-network library. Path planning — planners,
cost maps, route logic, robot middleware — belongs downstream, not here. Outputs should
be easy for a planner to consume, but the planning itself stays outside geobn. Flag
changes that pull it in.

**Documentation style.** In `README.md`, `docs/`, docstrings, and error messages, geobn
is described as if every feature had always been there. Flag before-and-after framing:
"now", "no longer", "previously", "new in this version", "the restriction has been
lifted", and similar. Flag marketing tone, punchy one-liners, rhetorical contrasts
("not X, but Y"), dramatic em-dash asides, and caveats stacked for emphasis. Plain
declarative sentences are the target. `CHANGELOG.md`, commit messages, and PR
descriptions are exempt, since describing a change is their job.

**Tests and changelog.** New behaviour should be covered by tests, and user-visible
changes should have a `CHANGELOG.md` entry.

## What not to flag

- **Style and formatting.** Ruff owns formatting, import order, and lint. Do not repeat
  it, and do not bikeshed naming that is already consistent with the module.
- **Backwards compatibility.** The library is pre-1.0 with no users. Breaking changes,
  missing deprecation paths, and absent compatibility shims are not findings.
- **Praise and summaries.** Do not open with what the PR does or close with an overall
  assessment. If there is nothing to report, say so in one line.
