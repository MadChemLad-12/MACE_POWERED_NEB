"""
==============================================================================
MACE MULTI-MODEL NEB COMPARISON — N models, ranked leaderboard
==============================================================================

PURPOSE:
  Run NEB simulations for Pt dissolution pathways (Pt, PtO, PtOH, PtO2,
  PtOH2) on solvated Pt(111) at varying O coverages using ANY NUMBER of
  MACE models simultaneously, and rank them against each other.

WORKFLOW:
  1. VALIDATE    — check that endpoints only differ by the expected
                   dissolving atoms (1 for Pt, 2 for PtO/PtOH2, 3 for
                   PtOH, 3 for PtO2... see SPECIES_MOVERS)
  2. NEB RUN     — run NEB with every model in MODELS; save outputs to
                   separate per-model directories
  3. COMPARE     — pairwise energy barriers, TS positions, path agreement
                   (RMSE), forces, for every pair of models
  4. PATHOLOGY   — flag per-image anomalies: energy spikes, force
                   divergence, atom displacement outliers, per model
  5. DFT POLISH  — once per config, take the REFERENCE_MODEL_KEY's path
                   and run a short VASP NEB "polish" — this becomes the
                   ground truth used to score every model
  6. LEADERBOARD — rank every model: vs DFT ground truth if available,
                   otherwise vs cross-model consensus (median profile)

EXPECTED CONFIG NAMES (parsed automatically):
  Close{coverage}Pt, Close{coverage}PtO, Close{coverage}PtOH,
  Close{coverage}PtO2, Close{coverage}PtOH2

USAGE:
  python neb_model_compare.py [--csv PATH] [--validate-only] [--no-neb]
                               [--dft-only] [--force-dft]
                               [--reference-model KEY] [--only-models a,b]
                               [--skip-models a,b]

==============================================================================
"""

import os
import re
import csv
import json
import time
import hashlib
import argparse
import warnings
import itertools
import numpy as np
import matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import BFGS, FIRE
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.units import Hartree, Bohr
from ase.mep.neb import NEB
from ase.mep import NEBTools
from mace.calculators import MACECalculator
from contextlib import contextmanager

# ==============================================================================
# TIMING
# ==============================================================================
# Cumulative wall-clock time per named phase. Populated via the `timed()`
# context manager below. Printed as a summary table at the end of the run
# and saved to reports/timings.json.
TIMINGS = {}


@contextmanager
def timed(phase_name: str):
    """
    Wrap a block of code and accumulate its wall-clock time under
    TIMINGS[phase_name]. Safe to call the same phase_name multiple times
    (e.g. once per config) — durations accumulate rather than overwrite.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        TIMINGS[phase_name] = TIMINGS.get(phase_name, 0.0) + dt
        print(f"  [⏱]  {phase_name}: {dt:.1f}s  (cumulative: {TIMINGS[phase_name]:.1f}s)")


def print_timing_summary():
    print("\n" + "=" * 70)
    print("  TIMING SUMMARY")
    print("=" * 70)
    total = TIMINGS.get("total_runtime", sum(v for k, v in TIMINGS.items() if k != "total_runtime"))
    for k, v in TIMINGS.items():
        if k == "total_runtime":
            continue
        pct = (v / total * 100) if total else 0.0
        print(f"  {k:<28}{v:>10.1f}s  ({pct:5.1f}%)")
    print(f"  {'─'*48}")
    print(f"  {'TOTAL':<28}{total:>10.1f}s")
    print()


def save_timing_summary():
    report_dir = os.path.join(OUTPUT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "timings.json")
    with open(path, "w") as f:
        json.dump(TIMINGS, f, indent=2)
    print(f"[✓] Timing summary saved: {path}")

# ==============================================================================
# SETTINGS — EDIT THESE
# ==============================================================================

# ── Models ────────────────────────────────────────────────────────────────────
# Add as many models as you like here — every key MUST be unique or earlier
# entries will be silently overwritten (this is what bit us before: two
# entries both named "finetuned" meant only the second one ever loaded).
MODELS = {
    "foundational": {
        "path":  "/home/user/Documents/Programs/Python/ASE/MACE/mace-mp-0b3-medium-float32.model",
        "label": "MACE-MP-0b3",
        "color": "#2196F3",
    },
    #"finetuned_v4": {
    #    "path":  "/home/user/Documents/Models/Simulations/MACE/neb/mace_V4_active_learning_stagetwo.model",
    #    "label": "MACE-V4",
    #    "color": "#F44336",
    #},
    "finetuned_v5": {
        "path":  "/home/user/Documents/Programs/Python/ASE/MACE/active_learning/mace_V5_active_learning_stagetwo.model",
        "label": "MACE-V5",
        "color": "#4CAF50",
    },
    # "finetuned_v6": {
    #     "path":  "/path/to/your/next/model.model",
    #     "label": "MACE-V6",
    #     "color": "#9C27B0",
    # },
}

# ── Reference model ────────────────────────────────────────────────────────
# Used for: (1) pre-relaxing NEB endpoints before interpolation, and
# (2) providing the initial-guess path for the DFT "polish" in Part 5.
# Pick whichever model you currently trust most as a starting point — this
# does NOT bias the leaderboard, since every model (including this one) is
# scored against the DFT result independently.
REFERENCE_MODEL_KEY = "finetuned_v5"

# ── Paths ─────────────────────────────────────────────────────────────────────
PATH_CSV    = "/home/user/Documents/Models/Simulations/MACE/neb/Pt_Diss_Neb_test.csv"
OUTPUT_ROOT = "neb_comparison"            # root output dir; sub-dirs per model

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda"
DTYPE  = "float32"

# ── ENDPOINT PRE-RELAXATION OPTIONS ─────────────────────────────────────────
RELAX_ENDPOINTS = True        # Set to True to optimize initial/final structures first
ENDPOINT_FMAX = 0.8         # Maximum residual force tolerance (eV/Å)

# How endpoints are relaxed before NEB interpolation. Overridable via
# --endpoint-relax-mode on the command line.
#   "per-model" (recommended, default) — each model relaxes its OWN copy of
#       the endpoints with its own calculator, so every model's NEB starts
#       from a geometry that's actually a minimum on ITS potential surface.
#       This avoids biasing the leaderboard/consensus metrics toward whichever
#       model is picked as REFERENCE_MODEL_KEY.
#   "shared" (legacy behaviour) — only REFERENCE_MODEL_KEY relaxes the
#       endpoints once; every model's NEB (including the reference model's)
#       starts from that single shared geometry. Kept for backward
#       compatibility / for deliberately testing "what if all models start
#       from the same place".
ENDPOINT_RELAX_MODE = "per-model"   # "per-model" | "shared"

# ── Endpoint relaxation caching ─────────────────────────────────────────────
# If True, before running FIRE on an endpoint we hash its pre-relaxation
# geometry (positions/numbers/cell) + model_key + tolerances, and check
# {OUTPUT_ROOT}/relax_cache/ for a previously-relaxed match. Saves the
# 30s-60s/endpoint relaxation cost on repeat runs (e.g. after fixing an
# unrelated bug downstream) — written to disk immediately after each
# relaxation finishes, so it's safe even if the run later crashes.
USE_RELAX_CACHE = True
RELAX_CACHE_DIR = os.path.join(OUTPUT_ROOT, "relax_cache")

# ── Structure validation ───────────────────────────────────────────────────────
# Maximum distance (Å) an atom must move to be counted as "dissolving"
DISSOLVING_THRESHOLD = 2.0      # atoms beyond this are flagged as movers
# Maximum distance an atom may move AND still be called "stationary"
STATIONARY_THRESHOLD = 0.8      # atoms beyond this but not dissolving → warning
# Whether to abort NEB if the validator finds the wrong number of movers
ABORT_ON_VALIDATION_FAIL = False

# ── NEB ───────────────────────────────────────────────────────────────────────
N_IMAGES      = 12
NEB_FMAX      = 0.8
NEB_OPTIMIZER = "FIRE"
NEB_MAX_STEPS = 800
CLIMB         = False
FIX_BY_HEIGHT = True
FIX_HEIGHT_THRESHOLD = 2.7      # Å — fix atoms below this z-height

# ── Pathology detection ───────────────────────────────────────────────────────
PATHOLOGY_ENERGY_SPIKE  = 2.0   # eV — flag image if E jumps > this vs neighbour
PATHOLOGY_FORCE_FMAX    = 5.0   # eV/Å — flag image if max force exceeds this
PATHOLOGY_ATOM_DISP     = 3.0   # Å — flag image if any atom moved > this vs init
PATHOLOGY_ENERGY_ABS    = 5.0   # eV above initial energy → absolute flag

# ── DFT refinement (ground truth for the leaderboard) ──────────────────────
RUN_DFT_REFINEMENT = False
DFT_TYPE           = "vasp"
DFT_COMMAND        = "vasp_std"
# Set True (or pass --force-dft) to re-run VASP NEB polish even if a cached
# {name}_dft_refined.extxyz already exists for that config. Default False
# means a config whose DFT polish already finished successfully is loaded
# straight from disk instead of resubmitting VASP.
FORCE_DFT          = False

# In future could set ibrion to 2 and nsw to 30 to do slight geo opts of each image. 
# I have removed 'nbands': 500, because of an error "I found NBANDS = 500, NELECT = 1258."
DFT_PARAMS = {
    'ibrion': -1, 'isif': 2, 'nsw': 0, 'ediffg': -0.05, 'prec': 'Accurate',
    'nelm': 150, 'ediff': 1e-6,  'ismear': -1, 'sigma': 0.1,
    'imix': 4, 'amix': 0.1, 'bmix': 1.0, 'gga': 'PE', 'ivdw': 11,
    'lcharg': False, 'lwave': False,
}

# ── CP2K (alternative DFT backend) ──────────────────────────────────────────
# Used instead of VASP when --dft-type cp2k is passed. Unlike the VASP path
# (which can run a live ASE-driven NEB polish via run_dft_path_refinement),
# CP2K support here is write-inputs / read-results only — i.e. designed for
# the same "write .inp files, hand them to the HPC scheduler, copy .out
# files back, read them in" workflow as --write-vasp-inputs/--read-vasp-results,
# just with CP2K's own input/output format. This matches how CP2K is
# actually run on HPC clusters (e.g. via SLURM), and reuses the exact
# template/parsing conventions from the existing CP2K active-learning
# pipeline so MACE training data and NEB-validation DFT stay consistent.
CP2K_COMMAND   = "cp2k.psmp"
CP2K_LIBDIR    = "/home/user/Documents/Models/CP2K/data"   # adjust to your CP2K lib path

# "sp" (default): single-point ENERGY_FORCE, same spirit as the VASP side's
#   ibrion=-1/nsw=0 — geometry untouched, just evaluate E/F once.
# "geo_opt": small CP2K-internal geometry relax (CP2K_GEO_OPT_MAX_STEPS
#   steps) before reporting final E/F — analogous to setting VASP's
#   ibrion=2/nsw=N. Set via --cp2k-run-type {sp,geo_opt}.
CP2K_RUN_TYPE        = "sp"
CP2K_GEO_OPT_MAX_STEPS = 50

HA_TO_EV          = Hartree              # eV per Hartree
HA_BOHR_TO_EV_ANG = Hartree / Bohr        # (eV/Å) per (a.u. force)

# {run_type} -> "ENERGY_FORCE" (sp) or "GEO_OPT" (geo_opt)
# {motion_block} -> "" (sp) or a &MOTION/&GEO_OPT block (geo_opt), inserted
# as a sibling of &FORCE_EVAL at the top level of the input file.
CP2K_TEMPLATE = """\
&GLOBAL
  PROJECT_NAME {name}
  RUN_TYPE {run_type}
  PRINT_LEVEL MEDIUM
&END GLOBAL
{motion_block}&FORCE_EVAL
  METHOD QS
  &DFT
    BASIS_SET_FILE_NAME {libdir}/BASIS_MOLOPT
    POTENTIAL_FILE_NAME {libdir}/GTH_POTENTIALS
    &MGRID
      CUTOFF 500
      REL_CUTOFF 50
      NGRIDS 5
    &END MGRID
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS
    &SCF
      SCF_GUESS ATOMIC
      MAX_SCF 150
      EPS_SCF 1.0E-6
      ADDED_MOS 500
      &SMEAR ON
        METHOD FERMI_DIRAC
        ELECTRONIC_TEMPERATURE [K] 1000
      &END SMEAR
      &MIXING
        METHOD BROYDEN_MIXING
        ALPHA 0.1
        BETA 1.0
        NBROYDEN 12
      &END MIXING
      &DIAGONALIZATION
        ALGORITHM STANDARD
      &END DIAGONALIZATION
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
      &vdW_POTENTIAL
        DISPERSION_FUNCTIONAL PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3
          REFERENCE_FUNCTIONAL PBE
          PARAMETER_FILE_NAME {libdir}/dftd3.dat
        &END PAIR_POTENTIAL
      &END vdW_POTENTIAL
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      ABC {a:.6f} {b:.6f} {c:.6f}
      PERIODIC XYZ
    &END CELL
    &COORD
{coords}
    &END COORD
{kinds}
  &END SUBSYS
  &PRINT
    &STRESS_TENSOR
    &END STRESS_TENSOR
    &FORCES
    &END FORCES
  &END PRINT
  STRESS_TENSOR ANALYTICAL
&END FORCE_EVAL
"""

CP2K_MOTION_GEO_OPT_BLOCK = """\
&MOTION
  &GEO_OPT
    OPTIMIZER BFGS
    MAX_ITER {max_steps}
    MAX_DR 0.003
    RMS_DR 0.0015
    MAX_FORCE 0.00045
    RMS_FORCE 0.0003
  &END GEO_OPT
&END MOTION
"""

CP2K_KIND_TEMPLATE = """\
    &KIND {symbol}
      BASIS_SET {basis}
      POTENTIAL {potential}
    &END KIND"""

CP2K_KIND_PARAMS = {
    "H":  ("DZVP-MOLOPT-SR-GTH-q1",  "GTH-PBE-q1"),
    "C":  ("DZVP-MOLOPT-SR-GTH-q4",  "GTH-PBE-q4"),
    "O":  ("DZVP-MOLOPT-SR-GTH-q6",  "GTH-PBE-q6"),
    "F":  ("DZVP-MOLOPT-SR-GTH-q7",  "GTH-PBE-q7"),
    "S":  ("DZVP-MOLOPT-SR-GTH-q6",  "GTH-PBE-q6"),
    "Pt": ("DZVP-MOLOPT-SR-GTH-q18", "GTH-PBE-q18"),
}


# ── Image ignore list ───────────────────────────────────────────────────────
# Lets you exclude specific NEB image indices from plots, barrier/RMSE
# calculations, and leaderboard scoring without re-running anything —
# e.g. a known-pathological image that you don't want polluting the
# comparison. Populated from --ignore-images at the CLI; format:
#   "*" key            -> applies to every config
#   a specific config name key -> applies only to that config
# An ignored image is NOT removed from the per-image breakdown CSV (it's
# still shown there, flagged), only from plots/metrics.
IGNORE_IMAGES = {}


# ==============================================================================
# SPECIES → EXPECTED MOVER COUNT
#   Derived from chemical formula of the dissolving species.
#   PtOH2 → Pt + O + H + H = 4 atoms dissolve? No — the SPECIES dissolves
#   together, so PtOH2 = 1 Pt + 1 O + 2 H = 4 atoms total.
#   Adjust these if your naming convention differs.
# ==============================================================================
SPECIES_MOVERS = {
    "Pt":    1,    # only the Pt atom dissolves
    "PtO":   2,    # Pt + O
    "PtOH":  3,    # Pt + O + H
    "PtO2":  3,    # Pt + O + O
    "PtOH2": 5,    # Pt + O + O + H + H
}

# ==============================================================================
# HELPERS
# ==============================================================================

def parse_species_from_name(name: str) -> str | None:
    """
    Extract the dissolving species from a config name like 'Close0.75PtOH'.
    Returns one of: Pt, PtO, PtOH, PtO2, PtOH2, or None if unrecognised.
    """
    for species in sorted(SPECIES_MOVERS.keys(), key=len, reverse=True):
        if species in name:
            return species
    return None


def get_fixed_indices(atoms):
    """Indices of atoms to freeze based on Z-height threshold."""
    if not FIX_BY_HEIGHT:
        return []
    return [a.index for a in atoms if a.position[2] < FIX_HEIGHT_THRESHOLD]


def mic_displacements(atoms_init, atoms_final):
    """
    Return per-atom displacement magnitudes (Å) between two structures,
    respecting minimum image convention for periodic boundary conditions.
    """
    cell = atoms_init.get_cell()
    pbc  = atoms_init.get_pbc()
    diffs, dists = find_mic(
        atoms_final.positions - atoms_init.positions,
        cell, pbc
    )
    return dists


def active_model_keys():
    """The model keys actually selected for this run (after --only/--skip filters)."""
    return list(MODELS.keys())


def parse_ignore_images(spec: str) -> dict:
    """
    Parse a --ignore-images CLI spec into {config_name_or_'*': set(indices)}.

    Formats (mix freely, separated by ';'):
      "0,13"                          -> ignore images 0 and 13 in every config
      "Close0.75PtO2:0,13"            -> ignore only in that config
      "Close0.75PtO2:0,13;Close0.5Pt:5" -> per-config rules for multiple configs
    """
    result = {}
    if not spec:
        return result
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            name, idxs = chunk.split(":", 1)
            name = name.strip()
        else:
            name, idxs = "*", chunk
        indices = {int(x.strip()) for x in idxs.split(",") if x.strip() != ""}
        result.setdefault(name, set()).update(indices)
    return result


def get_ignored_images(name: str) -> set:
    """Union of global ('*') and config-specific ignored image indices."""
    ignored = set(IGNORE_IMAGES.get("*", set()))
    ignored |= IGNORE_IMAGES.get(name, set())
    return ignored


def mask_ignored(name: str, values) -> np.ndarray:
    """Float array copy of `values` with NaN at any ignored image index for `name`."""
    arr = np.array(values, dtype=float)
    for idx in get_ignored_images(name):
        if 0 <= idx < len(arr):
            arr[idx] = np.nan
    return arr


def relative_to_first_finite(arr: np.ndarray) -> np.ndarray:
    """
    Like `arr - arr[0]`, but if arr[0] is NaN (e.g. image 0 got ignored),
    uses the first non-NaN entry as the reference instead of propagating
    NaN through the whole profile.
    """
    finite_idx = np.where(~np.isnan(arr))[0]
    if len(finite_idx) == 0:
        return arr
    return arr - arr[finite_idx[0]]


# ==============================================================================
# PART 1 — STRUCTURE VALIDATION
# ==============================================================================

def validate_endpoints(name: str, init_atoms, final_atoms) -> dict:
    """
    Check that only the expected number of atoms have moved significantly
    between the initial (surface-bound) and final (dissolved) endpoints.

    Returns a dict with keys:
      - passed (bool)
      - expected_movers (int)
      - species (str)
      - moving_atoms (list of dicts: index, symbol, displacement)
      - stationary_warnings (list of dicts)
      - message (str)
    """
    result = {
        "name":                name,
        "passed":              False,
        "expected_movers":     None,
        "species":             None,
        "moving_atoms":        [],
        "stationary_warnings": [],
        "message":             "",
    }

    # 1. Atom count
    if len(init_atoms) != len(final_atoms):
        result["message"] = (
            f"Atom count mismatch: init={len(init_atoms)}, "
            f"final={len(final_atoms)}"
        )
        return result

    # 2. Element order
    init_syms  = init_atoms.get_chemical_symbols()
    final_syms = final_atoms.get_chemical_symbols()
    mismatches = [(i, s1, s2) for i, (s1, s2) in enumerate(zip(init_syms, final_syms)) if s1 != s2]
    if mismatches:
        result["message"] = (
            f"Element mismatch at {len(mismatches)} index(es): "
            + ", ".join(f"idx {i}: {s1}→{s2}" for i, s1, s2 in mismatches[:5])
        )
        return result

    # 3. Per-atom displacements
    dists = mic_displacements(init_atoms, final_atoms)

    movers = []
    stat_warnings = []
    for i, (sym, d) in enumerate(zip(init_syms, dists)):
        if d > DISSOLVING_THRESHOLD:
            movers.append({"index": i, "symbol": sym, "displacement_A": float(d)})
        elif d > STATIONARY_THRESHOLD:
            stat_warnings.append({"index": i, "symbol": sym, "displacement_A": float(d)})

    result["moving_atoms"]        = movers
    result["stationary_warnings"] = stat_warnings

    # 4. Compare against expected count
    species = parse_species_from_name(name)
    result["species"] = species

    if species is None:
        result["message"] = (
            f"Could not parse dissolving species from name '{name}'. "
            f"Known: {list(SPECIES_MOVERS.keys())}"
        )
        # Don't fail hard — just warn
        result["passed"] = True
        return result

    expected = SPECIES_MOVERS[species]
    result["expected_movers"] = expected
    n_moving = len(movers)

    if n_moving == expected:
        result["passed"] = True
        result["message"] = (
            f"OK — {n_moving}/{expected} atoms moved (species: {species})"
        )
    else:
        result["passed"] = False
        result["message"] = (
            f"MISMATCH — found {n_moving} moving atoms, expected {expected} "
            f"for species '{species}'"
        )

    return result


def print_validation_report(vr: dict):
    """Pretty-print a single validation result."""
    status = "[✓]" if vr["passed"] else "[✗]"
    print(f"\n  {status} {vr['name']}: {vr['message']}")

    if vr["moving_atoms"]:
        print(f"      Moving atoms (>{DISSOLVING_THRESHOLD} Å):")
        for a in vr["moving_atoms"]:
            print(f"        Index {a['index']:>4}  {a['symbol']}  "
                  f"Δr = {a['displacement_A']:.3f} Å")

    if vr["stationary_warnings"]:
        print(f"      Stationary warnings ({STATIONARY_THRESHOLD}–{DISSOLVING_THRESHOLD} Å):")
        for a in vr["stationary_warnings"][:5]:
            print(f"        Index {a['index']:>4}  {a['symbol']}  "
                  f"Δr = {a['displacement_A']:.3f} Å")
        if len(vr["stationary_warnings"]) > 5:
            print(f"        ... and {len(vr['stationary_warnings'])-5} more.")


def run_all_validations(configs: list, verbose=True) -> list:
    """Validate all configs and return list of validation result dicts."""
    print("\n" + "="*70)
    print("  PART 1 — ENDPOINT STRUCTURE VALIDATION")
    print("="*70)

    all_results = []
    for config in configs:
        name = config["name"]
        init_path  = config["initial"]
        final_path = config["final"]

        if not os.path.exists(init_path):
            print(f"  [✗] {name}: initial file not found: {init_path}")
            all_results.append({"name": name, "passed": False,
                                 "message": "File not found: " + init_path})
            continue
        if not os.path.exists(final_path):
            print(f"  [✗] {name}: final file not found: {final_path}")
            all_results.append({"name": name, "passed": False,
                                 "message": "File not found: " + final_path})
            continue

        init_atoms  = read(init_path)
        final_atoms = read(final_path)
        vr = validate_endpoints(name, init_atoms, final_atoms)

        if verbose:
            print_validation_report(vr)

        all_results.append(vr)

    n_pass = sum(1 for r in all_results if r["passed"])
    n_fail = len(all_results) - n_pass
    print(f"\n  Validation summary: {n_pass} passed, {n_fail} failed\n")
    return all_results


# ==============================================================================
# PART 2 — MULTI-MODEL NEB
# ==============================================================================

def load_models() -> dict:
    """
    Load every MACE calculator in MODELS. A failure to load one model
    (bad path, corrupt file, OOM, etc.) is logged and that model is
    skipped rather than crashing the whole run — important for an
    unattended weekend job testing many candidates at once.
    """
    calcs = {}
    failed = []
    for key, cfg in MODELS.items():
        print(f"[→] Loading model '{cfg['label']}' ({key}) from: {cfg['path']}")
        try:
            calcs[key] = MACECalculator(
                model_paths=cfg["path"],
                device=DEVICE,
                default_dtype=DTYPE,
            )
            print(f"[✓] Loaded: {cfg['label']}")
        except Exception as e:
            print(f"[✗] FAILED to load '{cfg['label']}' ({key}): {e}")
            print(f"    Skipping this model for the rest of the run.")
            failed.append(key)

    if failed:
        print(f"\n[!] {len(failed)} model(s) failed to load and will be excluded: {failed}\n")

    if not calcs:
        raise RuntimeError("No MACE models loaded successfully — aborting.")

    if REFERENCE_MODEL_KEY not in calcs:
        fallback = next(iter(calcs.keys()))
        print(f"[!] REFERENCE_MODEL_KEY '{REFERENCE_MODEL_KEY}' did not load successfully. "
              f"Falling back to '{fallback}' as the reference model for endpoint "
              f"relaxation and DFT polishing.")
        globals()["REFERENCE_MODEL_KEY"] = fallback

    return calcs

def run_neb_single_model(
    init_atoms,
    final_atoms,
    calc,
    name: str,
    model_key: str,
    output_dir: str,
) -> dict:
    """
    Run NEB for a single model on a single config.

    Returns a result dict with energies, barriers, convergence info,
    and the list of image Atoms objects.
    """
    model_label = MODELS[model_key]["label"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  [→] NEB: {name}  |  model: {model_label}")

    result = {
        "name":        name,
        "model_key":   model_key,
        "model_label": model_label,
        "converged":   False,
        "images":      [],
        "energies":    [],
        "max_forces":  [],
        "barrier_forward":  None,
        "barrier_reverse":  None,
        "reaction_energy":  None,
        "ts_image_index":   None,
        "E_initial":        None,
        "E_final":          None,
        "E_ts":             None,
        "elapsed_s":        None,
        "error":            None,
    }

    # ── Build image list ──────────────────────────────────────────────────────
    images = [init_atoms.copy()]
    for _ in range(N_IMAGES):
        images.append(init_atoms.copy())
    images.append(final_atoms.copy())

    #neb = NEB(images, climb=CLIMB, allow_shared_calculator=True)
    neb = NEB(images, climb=CLIMB, allow_shared_calculator=True)
    neb.interpolate(apply_constraint=False)

    fixed_indices = get_fixed_indices(init_atoms)
    for image in images:
        if fixed_indices:
            image.set_constraint(FixAtoms(indices=fixed_indices))
        image.calc = calc

    # ── Optimise ──────────────────────────────────────────────────────────────
    traj_path = os.path.join(output_dir, f"{name}_neb.traj")
    log_path  = os.path.join(output_dir, f"{name}_neb.log")

    if NEB_OPTIMIZER == "FIRE":
        optimizer = FIRE(neb, trajectory=traj_path, logfile=log_path)
    else:
        optimizer = BFGS(neb, trajectory=traj_path, logfile=log_path)

    t0 = time.perf_counter()
    try:
        converged = optimizer.run(fmax=NEB_FMAX, steps=NEB_MAX_STEPS)
        result["converged"] = converged
        print(f"    [{'✓' if converged else '~'}] NEB {'converged' if converged else 'did not converge'} "
              f"in {time.perf_counter()-t0:.1f} s")
    except Exception as e:
        result["error"] = str(e)
        result["elapsed_s"] = time.perf_counter() - t0
        print(f"    [✗] NEB failed: {e}")
        return result

    result["elapsed_s"] = time.perf_counter() - t0

    # ── Extract energies and forces ───────────────────────────────────────────
    energies   = []
    max_forces = []
    for image in images:
        try:
            e = image.get_potential_energy()
            f = np.sqrt((image.get_forces() ** 2).sum(axis=1)).max()
        except Exception:
            e, f = np.nan, np.nan
        energies.append(e)
        max_forces.append(f)

    result["images"]     = images
    result["energies"]   = energies
    result["max_forces"] = max_forces

    # ── Barrier analysis ──────────────────────────────────────────────────────
    # Ignored images (--ignore-images) are excluded here: they don't count as
    # TS candidates, and if image 0/-1 itself is ignored, the reference
    # endpoint shifts to the first/last non-ignored image instead. This is
    # the model-comparison side, so it stays masked — the DFT ground truth
    # in extract_dft_reference() deliberately does NOT mask, since the goal
    # is to stop a model's own bad endpoint geometry from inflating its
    # apparent barrier, not to throw away real DFT data.
    energies_masked = mask_ignored(name, energies)
    finite_idx = np.where(~np.isnan(energies_masked))[0]

    if len(finite_idx) >= 2:
        ref_first = energies_masked[finite_idx[0]]
        ref_last  = energies_masked[finite_idx[-1]]
        intermediate = [(i, energies_masked[i]) for i in range(1, len(energies_masked) - 1)
                        if not np.isnan(energies_masked[i])]

        if intermediate:
            ts_idx, E_ts = max(intermediate, key=lambda x: x[1])
            result["E_initial"]       = ref_first
            result["E_final"]         = ref_last
            result["E_ts"]            = E_ts
            result["ts_image_index"]  = ts_idx
            result["barrier_forward"] = E_ts - ref_first
            result["barrier_reverse"] = E_ts - ref_last
            result["reaction_energy"] = ref_last - ref_first

            print(f"    E_barrier (fwd) = {result['barrier_forward']:+.4f} eV  "
                  f"|  ΔE = {result['reaction_energy']:+.4f} eV  "
                  f"|  TS @ image {ts_idx}"
                  f"{'  (ignoring images ' + str(sorted(get_ignored_images(name))) + ')' if get_ignored_images(name) else ''}")

    # ── Save extxyz ───────────────────────────────────────────────────────────
    # IMPORTANT: we attach each image's actual energy/forces via a
    # SinglePointCalculator before writing. Previously this block *stripped*
    # energy/free_energy/forces/energies from every image before saving — so
    # the extxyz on disk never actually had any calculated values in it, and
    # any later reload (e.g. --no-neb) silently got NaNs back out. Attaching
    # a SinglePointCalculator makes ASE's extxyz writer persist real numbers
    # into the "energy"/"free_energy" info keys and the "forces" array,
    # which is exactly what load_saved_neb_result() reads back via
    # get_potential_energy()/get_forces().
    for idx, image in enumerate(images):
        try:
            e_img = image.get_potential_energy()
        except Exception:
            e_img = None
        try:
            f_img = image.get_forces()
        except Exception:
            f_img = None
        spc_energy = None if (e_img is None or (isinstance(e_img, float) and np.isnan(e_img))) else float(e_img)
        image.calc = SinglePointCalculator(image, energy=spc_energy, forces=f_img)
        
        # BAKE THE VALUES HERE:
        if e_img is not None:
            image.info[f"energy_{model_key}"] = spc_energy       # e.g., energy_finetuned_v5
        if f_img is not None:
            image.arrays[f"forces_{model_key}"] = f_img
            
        image.info["system_type"]  = name
        image.info["neb_image"]    = idx
        image.info["model"]        = model_label
        image.info["source"]       = "mace_neb_compare"

    xyz_path = os.path.join(output_dir, f"{name}_neb.extxyz")
    write(xyz_path, images, format="extxyz")

    # ── Plot individual energy profile ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        nebtools = NEBTools(images)
        nebtools.plot_band(ax=ax)
    except Exception:
        E_rel = np.array(energies) - energies[0]
        ax.plot(range(len(E_rel)), E_rel, "o-")
        ax.set_ylabel("ΔE (eV)")
    ax.set_title(f"{name} — {model_label}", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{name}_neb_profile.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return result

def load_saved_neb_result(model_key: str, name: str) -> dict | None:
    """
    Reload a previously-completed NEB run for (model_key, name) from its
    saved {name}_neb.extxyz file, reconstructing the same result dict shape
    that run_neb_single_model() returns — minus elapsed_s/convergence info,
    which aren't stored on disk.

    Used by --no-neb to resume straight into comparison/pathology/DFT
    refinement without recomputing any MACE NEB. Returns None if no saved
    file is found for this model/config pair.
    """
    model_label = MODELS[model_key]["label"]
    out_dir   = os.path.join(OUTPUT_ROOT, model_key, name)
    xyz_path  = os.path.join(out_dir, f"{name}_neb.extxyz")

    if not os.path.exists(xyz_path):
        return None

    try:
        images = read(xyz_path, index=":")
    except Exception as e:
        print(f"    [✗] Could not read saved NEB result for {model_label}/{name}: {e}")
        return None

    energies, max_forces = [], []
    for img in images:
        if f"energy_{model_key}" in img.info:
            e = img.info[f"energy_{model_key}"]
        else:
            try:
                e = img.get_potential_energy()
            except Exception:
                e = np.nan
                
        if f"forces_{model_key}" in img.arrays:
            f_arr = img.arrays[f"forces_{model_key}"]
            f = np.sqrt((f_arr ** 2).sum(axis=1)).max()
        else:
            try:
                f = np.sqrt((img.get_forces() ** 2).sum(axis=1)).max()
            except Exception:
                f = np.nan
        energies.append(e)
        max_forces.append(f)

    result = {
        "name": name, "model_key": model_key, "model_label": model_label,
        "converged": True,   # unknown from disk alone — assume prior run reported success
        "images": images, "energies": energies, "max_forces": max_forces,
        "barrier_forward": None, "barrier_reverse": None, "reaction_energy": None,
        "ts_image_index": None, "E_initial": None, "E_final": None, "E_ts": None,
        "elapsed_s": None, "error": None, "loaded_from_disk": True,
    }

    # Same masked barrier/TS logic as run_neb_single_model() — this resume
    # path must match it exactly, or --ignore-images silently has no effect
    # whenever results are reloaded from disk instead of freshly computed
    # (e.g. via --no-neb / --read-vasp-results / --read-dft-results).
    energies_masked = mask_ignored(name, energies)
    finite_idx = np.where(~np.isnan(energies_masked))[0]
    if len(finite_idx) >= 2:
        ref_first = energies_masked[finite_idx[0]]
        ref_last  = energies_masked[finite_idx[-1]]
        intermediate = [(i, energies_masked[i]) for i in range(1, len(energies_masked) - 1)
                        if not np.isnan(energies_masked[i])]
        if intermediate:
            ts_idx, E_ts = max(intermediate, key=lambda x: x[1])
            result.update({
                "E_initial": ref_first, "E_final": ref_last, "E_ts": E_ts,
                "ts_image_index": ts_idx,
                "barrier_forward": E_ts - ref_first,
                "barrier_reverse": E_ts - ref_last,
                "reaction_energy": ref_last - ref_first,
            })

    print(f"    [✓] Loaded saved NEB result: {model_label}/{name} "
          f"({len(images)} images from disk)"
          f"{'  (ignoring images ' + str(sorted(get_ignored_images(name))) + ')' if get_ignored_images(name) else ''}")
    return result


def structure_hash(atoms, extra: str = "") -> str:
    """
    Stable fingerprint of an Atoms object's pre-relaxation geometry +
    composition (+ optional context like model_key/tolerances), used as a
    cache key. Positions are rounded before hashing so float noise from a
    file write/re-read round-trip doesn't produce spurious cache misses.
    """
    h = hashlib.sha256()
    h.update(np.round(atoms.get_positions(), 5).tobytes())
    h.update(atoms.get_atomic_numbers().tobytes())
    if atoms.cell is not None:
        h.update(np.round(np.asarray(atoms.cell), 5).tobytes())
    h.update(extra.encode())
    return h.hexdigest()[:16]


def relax_with_cache(atoms, calc, model_key: str, fmax: float, steps: int,
                      traj_path: str | None = None, tag: str = "") -> tuple:
    """
    Relax `atoms` with FIRE, or reuse a cached result if this exact
    geometry has already been relaxed under this model_key/fmax/steps
    combination. Returns (relaxed_atoms, was_cached, elapsed_s).

    The cache key is computed from the geometry BEFORE optimisation, so a
    fresh read() of the same starting structure on a later run will still
    hit the cache. `tag` lets the caller fold in extra context (e.g. the
    config name) if two different configs could otherwise coincidentally
    hash the same way — not strictly needed since geometry+model_key is
    already very specific, but cheap insurance.
    """
    if not USE_RELAX_CACHE:
        atoms.calc = calc
        opt = FIRE(atoms, logfile=None, trajectory=traj_path)
        t0 = time.perf_counter()
        opt.run(fmax=fmax, steps=steps)
        return atoms, False, time.perf_counter() - t0

    os.makedirs(RELAX_CACHE_DIR, exist_ok=True)
    key = structure_hash(atoms, extra=f"{model_key}_{fmax}_{steps}_{tag}")
    cache_path = os.path.join(RELAX_CACHE_DIR, f"{key}.extxyz")

    if os.path.exists(cache_path):
        cached = read(cache_path)
        cached.calc = calc   # reattach a live calculator for downstream use (NEB needs forces)
        return cached, True, 0.0

    atoms.calc = calc
    opt = FIRE(atoms, logfile=None, trajectory=traj_path)
    t0 = time.perf_counter()
    opt.run(fmax=fmax, steps=steps)
    elapsed = time.perf_counter() - t0
    write(cache_path, atoms)
    return atoms, False, elapsed


def run_all_nebs(
    configs: list,
    calcs: dict,
    validation_results: list,
    no_neb: bool = False,
    endpoint_relax_mode: str = "per-model",
) -> dict:
    """
    Run NEB for every config × every loaded model.
    Each model run is wrapped defensively so that one model crashing on
    one config (e.g. a NaN energy, a calculator error) doesn't stop the
    rest of an unattended multi-model, multi-config sweep.

    If no_neb=True, no relaxation and no NEB optimisation is performed at
    all — instead each model/config's previously-saved result is reloaded
    from {OUTPUT_ROOT}/{model_key}/{name}/{name}_neb.extxyz. This is the
    "resume" path: useful when MACE NEB already finished successfully in a
    prior run and you only need to redo something downstream (DFT polish,
    pathology detection, leaderboard) — e.g. after fixing a VASP POTCAR
    path that caused Part 5 to fail last time.
    """
    print("\n" + "="*70)
    print("  PART 2 — MULTI-MODEL NEB SIMULATIONS")
    print(f"  Models in this run: {', '.join(MODELS[k]['label'] for k in calcs.keys())}")
    if no_neb:
        print(f"  Mode: --no-neb — reloading saved results from disk, no computation.")
    else:
        print(f"  Endpoint relax mode: {endpoint_relax_mode}")
    print("="*70)

    # Build a quick lookup for validation pass/fail
    val_lookup = {r["name"]: r for r in validation_results}

    neb_results = {}
    for config in configs:
        name       = config["name"]
        init_path  = config["initial"]
        final_path = config["final"]

        vr = val_lookup.get(name, {})
        if ABORT_ON_VALIDATION_FAIL and not vr.get("passed", True):
            print(f"\n  [!] Skipping NEB for {name}: failed validation.")
            continue

        # ── RESUME PATH: load everything from disk, skip relax + NEB entirely ──
        if no_neb:
            neb_results[name] = {}
            with timed("resume_load_from_disk"):
                for model_key in calcs.keys():
                    loaded = load_saved_neb_result(model_key, name)
                    if loaded is not None:
                        neb_results[name][model_key] = loaded
                    else:
                        print(f"    [!] No saved result on disk for "
                              f"{MODELS[model_key]['label']}/{name} — skipping.")
            continue

        if not (os.path.exists(init_path) and os.path.exists(final_path)):
            print(f"\n  [!] Skipping NEB for {name}: missing structure file(s).")
            continue

        # ── ENDPOINT RELAXATION ──────────────────────────────────────────────
        # Either one shared relaxation (legacy "shared" mode, using only
        # REFERENCE_MODEL_KEY) or one relaxation per model ("per-model" mode,
        # default) so each model's NEB starts from ITS OWN relaxed minimum.
        endpoints_by_model = {}   # model_key -> (init_atoms, final_atoms)

        if RELAX_ENDPOINTS:
            relax_targets = (
                {REFERENCE_MODEL_KEY: calcs[REFERENCE_MODEL_KEY]}
                if endpoint_relax_mode == "shared" else calcs
            )
            with timed("endpoint_relaxation"):
                for model_key, target_calc in relax_targets.items():
                    init_atoms  = read(init_path)
                    final_atoms = read(final_path)
                    print(f"\n      [→] Pre-relaxing endpoints for {name} via "
                          f"{MODELS[model_key]['label']} "
                          f"({'reference model, shared' if endpoint_relax_mode == 'shared' else 'own model'})...")

                    for label, atoms in [("Initial", init_atoms), ("Final", final_atoms)]:
                        traj_path = os.path.join(OUTPUT_ROOT, f"{name}_{model_key}_{label}_relax.traj")
                        atoms, was_cached, t_opt = relax_with_cache(
                            atoms, target_calc, model_key,
                            fmax=ENDPOINT_FMAX, steps=800,
                            traj_path=traj_path, tag=f"{name}_{label}",
                        )
                        if label == "Initial":
                            init_atoms = atoms
                        else:
                            final_atoms = atoms

                        f_max = np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max()
                        if was_cached:
                            print(f"      [✓] {label}: cache hit — reused prior relaxation. "
                                  f"Max force: {f_max:.4f} eV/Å")
                        else:
                            print(f"      [✓] {label} optimized ({t_opt:.1f}s). "
                                  f"Max force: {f_max:.4f} eV/Å")

                    endpoints_by_model[model_key] = (init_atoms, final_atoms)

                    out_ep = os.path.join(OUTPUT_ROOT, "relaxed_endpoints")
                    os.makedirs(out_ep, exist_ok=True)
                    tag = model_key if endpoint_relax_mode == "per-model" else "shared"
                    write(os.path.join(out_ep, f"{name}_{tag}_initial_relaxed.xyz"), init_atoms)
                    write(os.path.join(out_ep, f"{name}_{tag}_final_relaxed.xyz"),   final_atoms)

            print(f"      [✓] Saved relaxed endpoints to: {os.path.join(OUTPUT_ROOT, 'relaxed_endpoints')}/")

            if endpoint_relax_mode == "shared":
                # Every model reuses the single reference-model-relaxed geometry.
                shared_pair = endpoints_by_model[REFERENCE_MODEL_KEY]
                for model_key in calcs.keys():
                    endpoints_by_model[model_key] = shared_pair
        else:
            # No relaxation at all — every model uses the raw input structures.
            for model_key in calcs.keys():
                endpoints_by_model[model_key] = (read(init_path), read(final_path))

        print(f"      [✓] Endpoints stabilized. Proceeding to multi-model NEB.")
        neb_results[name] = {}
        for model_key, calc in calcs.items():
            init_atoms, final_atoms = endpoints_by_model[model_key]
            out_dir = os.path.join(OUTPUT_ROOT, model_key, name)
            try:
                with timed("neb_optimisation"):
                    result = run_neb_single_model(
                        init_atoms.copy(), final_atoms.copy(),
                        calc, name, model_key, out_dir,
                    )
            except Exception as e:
                print(f"    [✗] Unhandled error running {MODELS[model_key]['label']} "
                      f"on {name}: {e}. Skipping this model/config pair.")
                result = {
                    "name": name, "model_key": model_key,
                    "model_label": MODELS[model_key]["label"],
                    "converged": False, "images": [], "energies": [],
                    "max_forces": [], "barrier_forward": None,
                    "barrier_reverse": None, "reaction_energy": None,
                    "ts_image_index": None, "elapsed_s": None,
                    "error": str(e),
                }
            neb_results[name][model_key] = result

    return neb_results

# ==============================================================================
# PART 3 — MODEL COMPARISON (all pairs, N models)
# ==============================================================================

def compare_models(neb_results: dict) -> list:
    """
    For every config, compute pairwise comparison metrics between every
    pair of models that both produced usable energies. With N models this
    is C(N,2) pairs per config — useful for understanding which models
    agree/disagree, but NOT by itself a "best model" ranking (see Part 6
    leaderboard for that, which scores against DFT ground truth).
    """
    print("\n" + "="*70)
    print("  PART 3 — PAIRWISE MODEL COMPARISON")
    print("="*70)

    if IGNORE_IMAGES==True:
        print("Note: This section will not ignore your images!!!")
    keys = active_model_keys()
    if len(keys) < 2:
        print("  [!] Need at least 2 models to compare.")
        return []

    comparisons = []

    for name, model_results in neb_results.items():
        available = [k for k in keys
                     if k in model_results
                     and not model_results[k].get("error")
                     and model_results[k].get("energies")]

        if len(available) < 2:
            continue

        print(f"\n  {'─'*60}")
        print(f"  {name}  (pairwise RMSE vs each other, eV)")
        print(f"  {'─'*60}")

        for k1, k2 in itertools.combinations(available, 2):
            r1 = model_results[k1]
            r2 = model_results[k2]

            E1 = np.array(r1["energies"])
            E2 = np.array(r2["energies"])
            F1 = np.array(r1["max_forces"])
            F2 = np.array(r2["max_forces"])

            E1_rel = E1 - E1[0]
            E2_rel = E2 - E2[0]

            n = min(len(E1), len(E2))
            energy_rmse = float(np.sqrt(np.mean((E1_rel[:n] - E2_rel[:n])**2)))
            energy_mae  = float(np.mean(np.abs(E1_rel[:n] - E2_rel[:n])))
            force_rmse  = float(np.sqrt(np.nanmean((F1[:n] - F2[:n])**2)))

            bf1, bf2 = r1["barrier_forward"], r2["barrier_forward"]
            br1, br2 = r1["barrier_reverse"], r2["barrier_reverse"]
            re1, re2 = r1["reaction_energy"], r2["reaction_energy"]
            ts1, ts2 = r1["ts_image_index"], r2["ts_image_index"]

            barrier_pct = None
            if bf1 is not None and bf2 is not None and bf1 != 0:
                barrier_pct = 100.0 * (bf2 - bf1) / abs(bf1)

            comp = {
                "name":                 name,
                "model_a_key":          k1,
                "model_b_key":          k2,
                "model_a":              MODELS[k1]["label"],
                "model_b":              MODELS[k2]["label"],
                "barrier_forward_a":    bf1,
                "barrier_forward_b":    bf2,
                "barrier_forward_diff": (bf2 - bf1) if (bf1 is not None and bf2 is not None) else None,
                "barrier_reverse_a":    br1,
                "barrier_reverse_b":    br2,
                "reaction_energy_a":    re1,
                "reaction_energy_b":    re2,
                "ts_image_a":           ts1,
                "ts_image_b":           ts2,
                "energy_rmse_eV":       energy_rmse,
                "energy_mae_eV":        energy_mae,
                "max_force_rmse_eVA":   force_rmse,
                "barrier_pct_diff":     barrier_pct,
                "energies_a":           E1_rel.tolist(),
                "energies_b":           E2_rel.tolist(),
                "max_forces_a":         F1.tolist(),
                "max_forces_b":         F2.tolist(),
            }
            comparisons.append(comp)

            bf1s = f"{bf1:+.3f}" if bf1 is not None else "N/A"
            bf2s = f"{bf2:+.3f}" if bf2 is not None else "N/A"
            print(f"    {MODELS[k1]['label']:<14} vs {MODELS[k2]['label']:<14}  "
                  f"ΔE‡: {bf1s} / {bf2s} eV   E_RMSE={energy_rmse:.4f}  "
                  f"F_RMSE={force_rmse:.4f}")

    return comparisons



def plot_comparison(comparisons: list, neb_results: dict):
    """Per-config overlay of every model's energy/force profile (scales to N models)."""
    cmp_dir = os.path.join(OUTPUT_ROOT, "comparison_plots")
    os.makedirs(cmp_dir, exist_ok=True)

    for name, model_results in neb_results.items():
        available = [k for k in active_model_keys()
                     if k in model_results and model_results[k].get("energies")]
        if not available:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Model comparison — {name}", fontsize=13, fontweight="bold")

        ax = axes[0]
        for k in available:
            r = model_results[k]
            E = np.array(r["energies"])
            c = MODELS[k]["color"]
            lbl = MODELS[k]["label"]
            ax.plot(range(len(E)), E - E[0], "o-", color=c, label=lbl,
                    linewidth=2, markersize=5)
        ax.set_xlabel("NEB image")
        ax.set_ylabel("Relative energy (eV)")
        ax.set_title("Energy profile")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        ax = axes[1]
        for k in available:
            r = model_results[k]
            F = r["max_forces"]
            c = MODELS[k]["color"]
            lbl = MODELS[k]["label"]
            ax.plot(range(len(F)), F, "s--", color=c, label=lbl,
                    linewidth=1.5, markersize=4, alpha=0.8)
        ax.axhline(NEB_FMAX, color="gray", linestyle=":", linewidth=1,
                   label=f"NEB F_max threshold ({NEB_FMAX} eV/Å)")
        ax.set_xlabel("NEB image")
        ax.set_ylabel("Max force (eV/Å)")
        ax.set_title("Max force per image")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(cmp_dir, f"{name}_comparison.png"), dpi=150)
        plt.close()

    plot_barrier_summary(neb_results, cmp_dir)
    print(f"[✓] Comparison plots saved to: {cmp_dir}/")


def plot_barrier_summary(neb_results: dict, cmp_dir: str):
    """Grouped bar chart: one bar per model per config (scales to N models)."""
    names = list(neb_results.keys())
    keys  = active_model_keys()
    if not names or not keys:
        return

    x = np.arange(len(names))
    width = 0.8 / max(len(keys), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.5), 5))

    for j, k in enumerate(keys):
        barriers = []
        for name in names:
            r = neb_results[name].get(k, {})
            bf = r.get("barrier_forward")
            barriers.append(bf if bf is not None else 0.0)
        ax.bar(x + j*width, barriers, width, label=MODELS[k]["label"],
               color=MODELS[k]["color"], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width*(len(keys)-1)/2)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Forward barrier (eV)")
    ax.set_title("Forward dissolution barriers — all models")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(cmp_dir, "summary_barriers.png"), dpi=150)
    plt.close()
    print(f"[✓] Summary barrier plot saved ({len(keys)} models).")


# ==============================================================================
# PART 4 — NEB PATHOLOGY DETECTION (per model, scales to N models)
# ==============================================================================

def detect_pathologies(neb_results: dict) -> dict:
    """
    Scan every NEB trajectory (per model, per config) for signs that the
    MACE model is performing badly.

    Flags per image:
      - ENERGY_SPIKE : energy jump > PATHOLOGY_ENERGY_SPIKE eV vs prior image
      - ENERGY_ABS   : absolute energy > E_initial + PATHOLOGY_ENERGY_ABS eV
      - FORCE_HIGH   : max force > PATHOLOGY_FORCE_FMAX eV/Å
      - ATOM_DISP    : any atom displaced > PATHOLOGY_ATOM_DISP Å from image 0

    Returns nested dict: pathologies[config_name][model_key] = list of flags
    """
    print("\n" + "="*70)
    print("  PART 4 — NEB PATHOLOGY DETECTION")
    print("="*70)

    all_pathologies = {}
    any_found = False

    for name, model_results in neb_results.items():
        all_pathologies[name] = {}

        for model_key, result in model_results.items():
            model_label = MODELS[model_key]["label"]
            images      = result.get("images", [])
            energies    = result.get("energies", [])
            max_forces  = result.get("max_forces", [])

            if not images or not energies:
                all_pathologies[name][model_key] = []
                continue

            flags = []
            E0    = energies[0]
            init_positions = images[0].positions.copy()
            cell  = images[0].get_cell()
            pbc   = images[0].get_pbc()

            for i, (image, E, Fmax) in enumerate(zip(images, energies, max_forces)):

                image_flags = []

                # ENERGY_SPIKE — sudden jump between consecutive images
                if i > 0:
                    prev_E = energies[i - 1]
                    if not (np.isnan(E) or np.isnan(prev_E)):
                        if abs(E - prev_E) > PATHOLOGY_ENERGY_SPIKE:
                            image_flags.append({
                                "type": "ENERGY_SPIKE",
                                "value": float(E - prev_E),
                                "threshold": PATHOLOGY_ENERGY_SPIKE,
                                "detail": f"ΔE = {E-prev_E:+.3f} eV vs image {i-1}",
                            })

                # ENERGY_ABS — implausibly high absolute energy
                if not np.isnan(E) and (E - E0) > PATHOLOGY_ENERGY_ABS:
                    image_flags.append({
                        "type": "ENERGY_ABS",
                        "value": float(E - E0),
                        "threshold": PATHOLOGY_ENERGY_ABS,
                        "detail": f"E − E_init = {E-E0:+.3f} eV",
                    })

                # FORCE_HIGH — forces far above NEB convergence
                if not np.isnan(Fmax) and Fmax > PATHOLOGY_FORCE_FMAX:
                    image_flags.append({
                        "type": "FORCE_HIGH",
                        "value": float(Fmax),
                        "threshold": PATHOLOGY_FORCE_FMAX,
                        "detail": f"F_max = {Fmax:.3f} eV/Å",
                    })

                # ATOM_DISP — any atom moved far from its image-0 position
                if i > 0:
                    _, dists = find_mic(
                        image.positions - init_positions,
                        cell, pbc,
                    )
                    max_disp_atom = int(np.argmax(dists))
                    max_disp      = float(dists[max_disp_atom])
                    sym           = image.get_chemical_symbols()[max_disp_atom]
                    if max_disp > PATHOLOGY_ATOM_DISP:
                        image_flags.append({
                            "type":      "ATOM_DISP",
                            "value":     max_disp,
                            "threshold": PATHOLOGY_ATOM_DISP,
                            "detail":    f"Atom {max_disp_atom} ({sym}) Δr = {max_disp:.2f} Å from image 0",
                        })

                if image_flags:
                    flags.append({
                        "structure": name,
                        "model":     model_label,
                        "image":     i,
                        "flags":     image_flags,
                    })
                    any_found = True

            all_pathologies[name][model_key] = flags

    # ── Pretty-print all flagged images ───────────────────────────────────────
    if not any_found:
        print("\n  [✓] No pathologies detected across all NEBs.\n")
    else:
        print("\n  [!] Pathologies detected:\n")
        for name, by_model in all_pathologies.items():
            for model_key, flags in by_model.items():
                if not flags:
                    continue
                model_label = MODELS[model_key]["label"]
                for f in flags:
                    for flag in f["flags"]:
                        print(f"  ⚠  {name}  |  {model_label}  |  image {f['image']:>2}  "
                              f"|  {flag['type']:<15}  {flag['detail']}")
        print()

    return all_pathologies


def plot_pathology_summary(all_pathologies: dict, neb_results: dict):
    """
    For each config × model, plot energy profile with pathological images
    highlighted so problems are immediately visible.
    """
    path_dir = os.path.join(OUTPUT_ROOT, "pathology_plots")
    os.makedirs(path_dir, exist_ok=True)
    keys = active_model_keys()
    
    for name, by_model in all_pathologies.items():
        has_anything = any(flags for flags in by_model.values())
        if not has_anything:
            continue

        n_models = max(len(keys), 1)
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=False)
        if n_models == 1:
            axes = [axes]

        fig.suptitle(f"Pathology map — {name}", fontsize=13, fontweight="bold")

        for ax, model_key in zip(axes, keys):
            flags       = by_model.get(model_key, [])
            model_label = MODELS[model_key]["label"]
            color       = MODELS[model_key]["color"]
            r           = neb_results[name].get(model_key, {})
            energies    = r.get("energies", [])

            if not energies:
                ax.set_title(f"{model_label} (no data)")
                continue

            E_rel = np.array(energies) - energies[0]
            ax.plot(range(len(E_rel)), E_rel, "o-", color=color,
                    linewidth=2, markersize=5, label="Energy profile")

            # Collect flagged image indices and their worst flag type
            flag_priority = {"ENERGY_ABS": 4, "ENERGY_SPIKE": 3,
                             "FORCE_HIGH": 2, "ATOM_DISP": 1}
            flagged_images = {}
            for f in flags:
                idx = f["image"]
                for flag in f["flags"]:
                    current_priority = flag_priority.get(flag["type"], 0)
                    if idx not in flagged_images or current_priority > flagged_images[idx][1]:
                        flagged_images[idx] = (flag["type"], current_priority, flag["detail"])

            flag_colors = {
                "ENERGY_ABS":   "#FF0000",
                "ENERGY_SPIKE": "#FF6600",
                "FORCE_HIGH":   "#9C27B0",
                "ATOM_DISP":    "#FF9800",
            }
            for idx, (ftype, _, detail) in flagged_images.items():
                if idx < len(E_rel):
                    fc = flag_colors.get(ftype, "red")
                    ax.scatter(idx, E_rel[idx], s=120, color=fc, zorder=5,
                               label=f"Image {idx}: {ftype}")
                    ax.annotate(f"img {idx}\n{ftype}",
                                xy=(idx, E_rel[idx]),
                                xytext=(idx + 0.3, E_rel[idx] + 0.05),
                                fontsize=7, color=fc)

            ax.set_xlabel("NEB image")
            ax.set_ylabel("ΔE (eV)")
            ax.set_title(f"{model_label}")
            ax.grid(True, linestyle="--", alpha=0.4)

            # Deduplicate legend
            handles, lbls = ax.get_legend_handles_labels()
            by_lbl = dict(zip(lbls, handles))
            ax.legend(by_lbl.values(), by_lbl.keys(), fontsize=7)

        plt.tight_layout()
        out_path = os.path.join(path_dir, f"{name}_pathology.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[✓] Pathology plot: {out_path}")
# ==============================================================================
# PART 5 — DFT NEB PATHWAY REFINEMENT (ground truth, once per config)
# ==============================================================================

def load_saved_dft_result(name: str) -> list | None:
    """
    Reload a previously-completed DFT polish for config `name` from
    {OUTPUT_ROOT}/dft_refinement/{name}/{name}_dft_refined.extxyz, if it
    exists. Returns the list of refined Atoms images (with energy/forces
    already attached via SinglePointCalculator), or None if nothing is
    saved yet for this config.
    """
    cfg_dft_root = os.path.join(OUTPUT_ROOT, "dft_refinement", name)
    xyz_path = os.path.join(cfg_dft_root, f"{name}_dft_refined.extxyz")
    if not os.path.exists(xyz_path):
        return None
    try:
        images = read(xyz_path, index=":")
    except Exception as e:
        print(f"    [✗] Could not read saved DFT result for {name}: {e}")
        return None
    print(f"    [✓] Loaded cached DFT refinement for '{name}' from {xyz_path} "
          f"({len(images)} images) — skipping VASP.")
    return images


def _reattach_endpoint_calcs(refined_images: list, source_images: list):
    """
    Endpoints (image 0 and -1) are never sent to VASP — they're meant to
    keep their original MACE-derived energy/forces. But Atoms.copy()
    deliberately strips any attached calculator, so after
    `[img.copy() for img in source_images]` the endpoints end up with NO
    calculator at all (not even the MACE one), which is what produced the
    "Atoms object has no calculator" warnings for image 0 and the last
    image. This re-pulls energy/forces from the original (un-copied)
    endpoint images and reattaches a SinglePointCalculator so endpoints
    carry real values through save/extract/plot.
    """
    for idx in (0, -1):
        try:
            e = source_images[idx].get_potential_energy()
            f = source_images[idx].get_forces()
        except Exception as err:
            print(f"      [~] Warning: endpoint image {idx} has no usable MACE "
                  f"energy/forces to carry over: {err}")
            continue
        refined_images[idx].calc = SinglePointCalculator(refined_images[idx], energy=e, forces=f)


def save_dft_result(name: str, refined_images: list):
    """
    Persist a finished DFT polish to disk as an extxyz with real
    energies/forces attached, so future runs (or --dft-only re-runs) can
    reload it via load_saved_dft_result() instead of resubmitting VASP.
    """
    cfg_dft_root = os.path.join(OUTPUT_ROOT, "dft_refinement", name)
    os.makedirs(cfg_dft_root, exist_ok=True)
    xyz_path = os.path.join(cfg_dft_root, f"{name}_dft_refined.extxyz")

    to_write = []
    for idx, img in enumerate(refined_images):
        try:
            # 1. Extract raw potential energy and force matrices from the calculator
            e = img.get_potential_energy()
            f = img.get_forces()  # Array shape: (N_atoms, 3)
            
            # 2. Compute the scalar maximum force component (Max Force)
            max_f = float(np.sqrt((f ** 2).sum(axis=1)).max())
        except Exception as err:
            print(f"      [~] Warning: Could not extract properties for image {idx}: {err}")
            e = None
            f = None
            max_f = None

        # Create a deep copy to prevent mutating the structures in memory
        img_copy = img.copy()
        
        # Clean up types for ASE compatibility
        spc_energy = None if (e is None or (isinstance(e, float) and np.isnan(e))) else float(e)
        
        # 3. Attach a SinglePointCalculator so the properties are read natively by standard tools
        img_copy.calc = SinglePointCalculator(img_copy, energy=spc_energy, forces=f)
        
        # 4. Bake global properties explicitly into the .info metadata block
        if spc_energy is not None:
            img_copy.info["energy_VASP"] = spc_energy
        if max_f is not None:
            img_copy.info["max_force_VASP"] = max_f
            
        img_copy.info["system_type"] = name
        img_copy.info["neb_image"]   = idx
        img_copy.info["source"]      = "vasp_dft_refinement"
        
        to_write.append(img_copy)

    write(xyz_path, to_write, format="extxyz")
    print(f"    [✓] Saved DFT refinement result: {xyz_path}")

def run_dft_path_refinement(name: str, mace_images: list, target_model_label: str) -> list:
    """
    Takes the REFERENCE_MODEL_KEY's finalized MACE NEB pathway and runs a
    short localized DFT-NEB "polish" via VASP. Runs ONCE PER CONFIG (not
    once per model pair) — this is the ground truth every model gets
    scored against in the Part 6 leaderboard.

    If a cached result already exists on disk for this config (see
    save_dft_result/load_saved_dft_result) and FORCE_DFT is not set, the
    cached result is reloaded instead of resubmitting VASP — this is what
    lets you re-run downstream steps (leaderboard, plots) after fixing a
    bug without burning DFT walltime on configs that already finished.
    """
    if not FORCE_DFT:
        cached = load_saved_dft_result(name)
        if cached is not None:
            return cached

    if DFT_TYPE == "cp2k":
        print(f"    [✗] DFT_TYPE='cp2k' doesn't support the live ASE-driven NEB polish "
              f"('run' mode) — CP2K support here is write-inputs/read-results only "
              f"(matches your HPC submission workflow anyway). Use --write-vasp-inputs "
              f"to write CP2K .inp files, then --read-vasp-results once the .out files "
              f"are copied back.")
        return []

    from ase.calculators.vasp import Vasp
    from ase.optimize import MDMin
    
    print(f"\n  [→] DFT Refinement: Polishing pathway from '{target_model_label}' via VASP NEB...")
    
    if DFT_PARAMS['nsw'] == "0":
        print(f"VASP will perfrom 0 steps and only validate the MACE pathway")
    
    # Deep copy the MACE path to prevent altering the baseline MACE structures in memory
    refined_images = [img.copy() for img in mace_images]
    cfg_dft_root   = os.path.join(OUTPUT_ROOT, "dft_refinement", name)

    # Atoms.copy() strips calculators, so the endpoints (which never get a
    # VASP calc below — only the moving images do) need their MACE
    # energy/forces explicitly carried over or they'll have no calculator
    # at all once we try to read properties off them later.
    _reattach_endpoint_calcs(refined_images, mace_images)

    # Assign unique working folders to each moving image to avoid parallel file collisions
    for idx, atoms in enumerate(refined_images[1:-1], start=1):
        image_dir = os.path.join(cfg_dft_root, f"image_{idx:02d}")
        os.makedirs(image_dir, exist_ok=True)
        
        calc = Vasp(
            command=DFT_COMMAND,
            directory=image_dir,
            **DFT_PARAMS
        )
        atoms.calc = calc

    # Re-instantiate the NEB string context for ASE using the DFT-linked images
    # We match your script's settings (climb=False, k=0.1)
    dft_neb = NEB(refined_images, climb=False, k=0.1)
    
    traj_path = os.path.join(cfg_dft_root, f"{name}_dft_refined.traj")
    log_path  = os.path.join(cfg_dft_root, f"{name}_dft_refined.log")
    
    # Using MDMin to coordinate the overarching NEB image-string translations
    optimizer = MDMin(dft_neb, trajectory=traj_path, logfile=log_path)
    
    t0 = time.perf_counter()
    try:
        # We run the optimizer loop. It stops when the image forces drop below 
        # 0.05 eV/Å or when VASP hitting its internal NSW step cap cuts the loop.
        print(f"      Running VASP-driven string optimization (Max VASP NSW={DFT_PARAMS['nsw']})...")
        optimizer.run(fmax=0.05, steps=DFT_PARAMS['nsw'])
        print(f"    [✓] DFT path refinement finished in {time.perf_counter()-t0:.1f} s")
    except Exception as err:
        print(f"    [✗] VASP NEB refinement errored out: {err}")
        return []

    save_dft_result(name, refined_images)
    return refined_images


def extract_dft_reference(name: str, refined_images: list) -> dict:
    """
    Pull energies/barriers out of a refined DFT path for leaderboard scoring.

    Applies the SAME --ignore-images mask as the model side. This matters
    because the endpoints (image 0/-1) never actually go through VASP/CP2K —
    they just carry over whichever MACE model's energy was used as the
    NEB endpoint, mislabeled as "DFT". Comparing a model against that
    isn't a real DFT comparison, so when an endpoint is ignored it's
    dropped from both sides, keeping the comparison apples-to-apples.
    """
    if not refined_images:
        return None
    energies = []
    for idx, img in enumerate(refined_images):
        try:
            energies.append(img.get_potential_energy())
        except Exception as e:
            print(f"      [~] Could not read DFT energy for {name} image {idx}: {e}")
            energies.append(None)

    if any(e is None for e in energies):
        print(f"      [✗] {name}: missing energy for one or more images — "
              f"skipping DFT reference for this config.")
        return None

    energies_masked = mask_ignored(name, energies)
    finite_idx = np.where(~np.isnan(energies_masked))[0]
    if len(finite_idx) < 2:
        print(f"      [✗] {name}: fewer than 2 non-ignored images with energy — "
              f"skipping DFT reference for this config.")
        return None

    ref_first = energies_masked[finite_idx[0]]
    ref_last  = energies_masked[finite_idx[-1]]
    E_rel = relative_to_first_finite(energies_masked)

    intermediate = [(i, energies_masked[i]) for i in range(1, len(energies_masked) - 1)
                    if not np.isnan(energies_masked[i])]
    if not intermediate:
        return None
    ts_idx, E_ts = max(intermediate, key=lambda x: x[1])

    return {
        "name":            name,
        "energies":        energies,           # raw, absolute, every image — for CSV/reporting
        "energies_rel":    E_rel.tolist(),      # NaN at ignored images
        "barrier_forward": E_ts - ref_first,
        "barrier_reverse": E_ts - ref_last,
        "reaction_energy": ref_last - ref_first,
        "ts_image_index":  ts_idx,
        "ignored_images":  sorted(get_ignored_images(name)),
    }


def plot_refined_dft_overlay(name: str, neb_results: dict, refined_images: list):
    """Plots the DFT-polished pathway alongside every model's NEB profile."""
    if not refined_images:
        return

    plot_dir = os.path.join(OUTPUT_ROOT, "comparison_plots")
    fig, ax = plt.subplots(figsize=(8, 5))

    for k in active_model_keys():
        r = neb_results.get(name, {}).get(k, {})
        E = r.get("energies")
        if not E:
            continue
        E_masked = mask_ignored(name, E)
        E_rel = relative_to_first_finite(E_masked)
        ax.plot(range(len(E_rel)), E_rel, "o-", color=MODELS[k]["color"],
                label=MODELS[k]["label"], alpha=0.6)

    try:
        dft_energies = [img.get_potential_energy() for img in refined_images]
        dft_masked = mask_ignored(name, dft_energies)
        dft_rel = relative_to_first_finite(dft_masked)
        ax.plot(range(len(dft_rel)), dft_rel, "D--", color="#111111",
                label="VASP Refined Path (ground truth)", linewidth=2.5, markersize=6)
    except Exception:
        pass

    ax.set_xlabel("NEB Image Index")
    ax.set_ylabel("Relative Energy (eV)")
    ax.set_title(f"All models vs. VASP-refined ground truth\nSystem: {name}", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name}_dft_refined_overlay.png"), dpi=150)
    plt.close()


def _resolve_cell(atoms, name: str):
    """
    Return (a, b, c) orthogonal cell lengths in Å for the CP2K &CELL block.
    CP2K's ABC keyword (as used in CP2K_TEMPLATE) only describes an
    orthogonal cell — if the actual ASE cell is non-orthogonal this will
    silently lose the off-diagonal components, so we warn loudly.
    """
    cell = atoms.get_cell()
    lengths = cell.lengths()
    if not np.allclose(cell.angles(), [90.0, 90.0, 90.0], atol=1e-3):
        print(f"      [!] {name}: cell is not orthogonal (angles={cell.angles()}) — "
              f"CP2K_TEMPLATE's ABC-only &CELL block will NOT capture this correctly. "
              f"Consider extending CP2K_TEMPLATE with a full A/B/C vector &CELL block.")
    a, b, c = lengths
    if a == 0 or b == 0 or c == 0:
        # No cell set at all — fall back to a padded bounding box so CP2K
        # at least gets something sane instead of a degenerate cell.
        pos = atoms.get_positions()
        span = pos.max(axis=0) - pos.min(axis=0)
        a, b, c = (span + 15.0)
        print(f"      [!] {name}: no cell set on Atoms — using padded bounding box "
              f"{a:.3f} {b:.3f} {c:.3f} Å. Set a real cell if this isn't intended.")
    return float(a), float(b), float(c)


def parse_stress_from_out(content: str):
    """
    Parse CP2K's analytical stress tensor block (printed because
    CP2K_TEMPLATE sets STRESS_TENSOR ANALYTICAL + &PRINT/&STRESS_TENSOR).
    Returns a 3x3 numpy array in GPa, or None if no stress block is found
    (e.g. STRESS_TENSOR wasn't enabled for this run).
    """
    block = re.search(
        r"STRESS\| Analytical stress tensor \[GPa\](.*?)STRESS\| 1/3 Trace",
        content, re.DOTALL
    )
    if not block:
        return None
    rows = []
    for line in block.group(1).strip().split("\n"):
        parts = line.split()
        nums = [float(p) for p in parts if re.match(r"^-?\d+\.\d+", p)]
        if len(nums) == 3:
            rows.append(nums)
    if len(rows) != 3:
        return None
    return np.array(rows)


def write_cp2k_sp(atoms, name: str, outdir) -> str:
    """Write one CP2K input file for `atoms` — single-point or small geo-opt,
    depending on the module-level CP2K_RUN_TYPE ('sp' or 'geo_opt')."""
    os.makedirs(outdir, exist_ok=True)
    a, b, c = _resolve_cell(atoms, name)

    coords = ""
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        coords += f"      {sym:<4s} {pos[0]:14.8f} {pos[1]:14.8f} {pos[2]:14.8f}\n"

    symbols_present = sorted(set(atoms.get_chemical_symbols()))
    kinds = "\n".join(
        CP2K_KIND_TEMPLATE.format(
            symbol=s, basis=CP2K_KIND_PARAMS[s][0], potential=CP2K_KIND_PARAMS[s][1]
        )
        for s in symbols_present if s in CP2K_KIND_PARAMS
    )
    missing = [s for s in symbols_present if s not in CP2K_KIND_PARAMS]
    if missing:
        print(f"      [!] No CP2K_KIND_PARAMS entry for: {missing} — they will be "
              f"absent from {name}.inp. Add them to CP2K_KIND_PARAMS before submitting.")

    if CP2K_RUN_TYPE == "geo_opt":
        run_type = "GEO_OPT"
        motion_block = CP2K_MOTION_GEO_OPT_BLOCK.format(max_steps=CP2K_GEO_OPT_MAX_STEPS)
    else:
        run_type = "ENERGY_FORCE"
        motion_block = ""

    inp = CP2K_TEMPLATE.format(
        name=name, libdir=CP2K_LIBDIR, a=a, b=b, c=c,
        coords=coords.rstrip(), kinds=kinds,
        run_type=run_type, motion_block=motion_block,
    )
    inp_file = Path(outdir) / f"{name}.inp"
    inp_file.write_text(inp)
    return str(inp_file)


def parse_cp2k_out(outfile_path, atoms):
    """
    Parse one CP2K .out file and return (energy_eV, forces, stress_voigt, ok).
    ok=True as long as energy was found — forces are optional. If forces
    can't be parsed (wrong count, format issue, etc.) we still return ok=True
    with forces=None so the energy at least makes it into the profile; the
    SinglePointCalculator caller handles None forces gracefully. ok=False
    only if energy itself couldn't be found or SCF didn't converge.
    A geo_opt run prints one ENERGY|/ATOMIC FORCES block per optimizer
    iteration — this always takes the LAST one (the converged/final
    geometry), not the first.
    """
    outfile = Path(outfile_path)
    if not outfile.exists():
        return None, None, None, False
    
    content = outfile.read_text()
    
    if "SCF run NOT converged" in content:
        print(f"      [!] {outfile.name}: SCF not converged — skipping.")
        return None, None, None, False
    
    # ── Energy ────────────────────────────────────────────────────────────────
    energy_matches = re.findall(
        r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[hartree\]\s+([-+]?\d+\.\d+)",
        content
    )
    
    if not energy_matches:
        # Not a CP2K QS output at all (e.g. vasp.out got through the filter)
        print(f"      [!] {outfile.name}: no CP2K energy line found — "
              f"wrong file type or incomplete run.")
        return None, None, None, False
    
    energy_eV = float(energy_matches[-1]) * HA_TO_EV
    
    # ── Forces ────────────────────────────────────────────────────────────────
    # CP2K force lines: "    idx  kind  symbol  fx  fy  fz"  (6 fields)
    # A header line "# Atom  Kind  Element  X  Y  Z" is also 6 fields but
    # its numeric columns aren't floats, so float() conversion fails safely.
    # I dont know why I cant get this force section to work
    force_blocks = re.findall(
        r"FORCES\| Atomic forces \[hartree/bohr\](.*?)SUM OF ATOMIC FORCES",
        content, re.DOTALL
    )
    
    forces = None
    if force_blocks:
        parsed = []
        for line in force_blocks[-1].strip().split("\n"):
            parts = line.split()
            if len(parts) == 6:
                try:
                    parsed.append([
                        float(parts[3]) * HA_BOHR_TO_EV_ANG,
                        float(parts[4]) * HA_BOHR_TO_EV_ANG,
                        float(parts[5]) * HA_BOHR_TO_EV_ANG
                    ])
                except ValueError:
                    pass   # header line — skip silently
        
        n_atoms = len(atoms) if atoms is not None else None
        if n_atoms is not None and len(parsed) == n_atoms:
            forces = np.array(parsed)
        elif n_atoms is not None and len(parsed) != n_atoms:
            print(f"      [!] {outfile.name}: force count mismatch "
                  f"(got {len(parsed)}, expected {n_atoms}) — using energy only.")
        elif n_atoms is None and parsed:
            forces = np.array(parsed)
        else:
            print(f"      [~] {outfile.name}: no ATOMIC FORCES block found — "
                  f"using energy only (forces not printed or run incomplete).")
    
    # ── Stress ────────────────────────────────────────────────────────────────
    stress_voigt = None
    stress = parse_stress_from_out(content)
    if stress is not None:
        stress_voigt = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
    
    # ok=True as long as we have energy — forces are a bonus, not a requirement
    return energy_eV, forces, stress_voigt, True


def write_vasp_inputs(name: str, mace_images: list) -> str:
    """
    Write VASP input files (POSCAR/INCAR/KPOINTS/POTCAR) for every moving
    image of the DFT-refinement pathway to
    {OUTPUT_ROOT}/dft_refinement/{name}/image_XX/, WITHOUT launching VASP.
    Endpoint images (0 and -1) are skipped — they never go through VASP.
    Mirrors the exact directory layout run_dft_path_refinement() uses so
    the resulting image_XX/ folders can be submitted directly to an HPC
    scheduler; copy the OUTCARs back into the same folders afterwards and
    re-run with --read-dft-results.
    """
    from ase.calculators.vasp import Vasp
    from ase.io import write as ase_write

    cfg_dft_root = os.path.join(OUTPUT_ROOT, "dft_refinement", name)
    written_dirs = []

    for idx, atoms in enumerate(mace_images[1:-1], start=1):
        image_dir = os.path.join(cfg_dft_root, f"image_{idx:02d}")
        os.makedirs(image_dir, exist_ok=True)

        calc = Vasp(directory=image_dir, **DFT_PARAMS)
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        try:
            calc.write_input(atoms_copy)
        except Exception as e:
            print(f"      [!] {name}/image_{idx:02d}: could not write VASP inputs: {e}")
            continue
        written_dirs.append(image_dir)

    manifest = {
        "name": name,
        "dft_type": "vasp",
        "vasp_nsw": DFT_PARAMS.get("nsw", 0),
        "n_images_total": len(mace_images),
        "n_moving_images": len(written_dirs),
        "moving_image_dirs": [os.path.basename(d) for d in written_dirs],
        "expected_output": "OUTCAR",
    }

    with open(os.path.join(cfg_dft_root, "vasp_inputs_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    mode_str = (f"geo_opt (nsw={DFT_PARAMS.get('nsw', 0)}, ibrion={DFT_PARAMS.get('ibrion', -1)})"
                if DFT_PARAMS.get("nsw", 0) > 0 else "single-point (nsw=0)")
    print(f"    [✓] Wrote VASP inputs ({mode_str}) for {len(written_dirs)} moving "
          f"image(s) of '{name}' to {cfg_dft_root}/image_XX/ — no VASP run was launched.")
    return cfg_dft_root

def _is_cp2k_output(path: Path) -> bool:
    """Quick check: does this .out file look like CP2K output rather than VASP?
    Reads only the first 50 lines to avoid loading huge files."""
    try:
        with open(path) as f:
            head = "".join(f.readline() for _ in range(50))
        return "CP2K|" in head or "PROGRAM STARTED" in head or "CP2K version" in head
    except Exception:
        return False


# Filenames that VASP commonly writes to the working directory — excluded
# from the CP2K output scanner so they don't get mistakenly parsed as CP2K.
_VASP_OUTPUT_NAMES = {
    "vasp.out", "vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR",
    "DOSCAR", "EIGENVAL", "XDATCAR", "PCDAT", "IBZKPT",
}


def _find_cp2k_output_files(image_dir: str):
    """
    Scan `image_dir` for CP2K output files rather than expecting a fixed
    naming convention. Returns (out_path, pos_xyz_path) where either may be
    None if not found. Handles the common naming variants:
        {name}_image_{idx:02d}.out   (written by write_cp2k_inputs)
        {name}_img{idx:02d}.out      (common HPC submission convention)
        any other single .out file in the directory
    Likewise for the geometry trajectory: {project}-pos-1.xyz.
    The project name is inferred from the .inp file if present, since CP2K
    always names its output files after PROJECT_NAME in &GLOBAL.

    Explicitly excludes known VASP output filenames so that mixed VASP+CP2K
    directories (common when both backends have been run) always pick the
    right file for the active --dft-type.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return None, None

    # Prefer the .out whose stem matches the .inp file (same PROJECT_NAME),
    # fall back to any single .out in the directory.
    inp_files = list(image_dir.glob("*.inp"))
    project_name = inp_files[0].stem if inp_files else None

    out_path = None
    if project_name:
        candidate = image_dir / f"{project_name}.out"
        if candidate.exists():
            out_path = candidate

    if out_path is None:
        out_candidates = [
            f for f in image_dir.glob("*.out")
            if f.name not in _VASP_OUTPUT_NAMES
            and not f.name.endswith(".err")
        ]
        if len(out_candidates) == 1:
            out_path = out_candidates[0]
        elif len(out_candidates) > 1:
            # Multiple candidates — prefer the one matching the .inp stem,
            # then prefer ones that look like CP2K output
            named = [f for f in out_candidates if project_name and f.stem == project_name]
            if named:
                out_path = named[0]
            else:
                cp2k_like = [f for f in out_candidates if _is_cp2k_output(f)]
                out_path = cp2k_like[0] if cp2k_like else out_candidates[0]
            print(f"      [~] {image_dir.name}: multiple .out candidates, using "
                  f"{out_path.name}")
        elif len(out_candidates) == 0 and not project_name:
            # No non-VASP .out found and no .inp to guide us — nothing to parse
            return None, None

    # CP2K writes the position trajectory as {PROJECT_NAME}-pos-1.xyz
    pos_path = None
    if out_path is not None:
        pos_candidate = image_dir / f"{out_path.stem}-pos-1.xyz"
        if pos_candidate.exists():
            pos_path = pos_candidate
        else:
            pos_candidates = list(image_dir.glob("*-pos-1.xyz"))
            if pos_candidates:
                pos_path = pos_candidates[0]

    return (str(out_path) if out_path else None,
            str(pos_path) if pos_path else None)


def read_vasp_outputs_from_disk(name: str, mace_images: list) -> list:
    """
    Scan {OUTPUT_ROOT}/dft_refinement/{name}/image_XX/ for OUTCAR files
    (e.g. copied back after running VASP on an HPC using the inputs from
    write_vasp_inputs()) and assemble them into the same refined_images
    shape run_dft_path_refinement() would have produced, so
    extract_dft_reference()/save_dft_result() work unchanged.

    Endpoints (image 0 and -1) are never sent to VASP — they keep their
    existing MACE-derived energy/forces. Any moving image whose OUTCAR
    isn't found yet (e.g. still queued on the HPC) is left at its MACE
    value and flagged, rather than failing the whole config.
    """
    cfg_dft_root = os.path.join(OUTPUT_ROOT, "dft_refinement", name)
    refined_images = [img.copy() for img in mace_images]

    # Same issue as run_dft_path_refinement: Atoms.copy() drops the
    # calculator, so endpoints need their MACE energy/forces reattached
    # explicitly — otherwise they end up with no calculator at all.
    _reattach_endpoint_calcs(refined_images, mace_images)

    n_expected = len(refined_images) - 2
    n_found = 0
    for idx in range(1, len(refined_images) - 1):
        image_dir = os.path.join(cfg_dft_root, f"image_{idx:02d}")
        outcar_path = os.path.join(image_dir, "OUTCAR")
        if not os.path.exists(outcar_path):
            print(f"      [!] {name}/image_{idx:02d}: no OUTCAR at {outcar_path} yet — "
                  f"keeping MACE value for this image.")
            continue
        try:
            vasp_atoms = read(outcar_path, index=-1)   # last ionic step in this OUTCAR
            e = vasp_atoms.get_potential_energy()
            f = vasp_atoms.get_forces()
        except Exception as err:
            print(f"      [✗] {name}/image_{idx:02d}: could not parse OUTCAR ({err}) — "
                  f"keeping MACE value for this image.")
            continue
        refined_images[idx].calc = SinglePointCalculator(refined_images[idx], energy=e, forces=f)
        n_found += 1

    status = "✓" if n_found == n_expected else "~"
    print(f"    [{status}] {name}: read {n_found}/{n_expected} moving-image OUTCAR(s) from disk.")
    if n_found == 0:
        return []
    return refined_images


def write_cp2k_inputs(name: str, mace_images: list) -> str:
    """
    CP2K counterpart to write_vasp_inputs(): writes one ENERGY_FORCE .inp
    file per moving image to {OUTPUT_ROOT}/dft_refinement/{name}/image_XX/
    — same directory layout as the VASP path, so the rest of the pipeline
    (manifests, read-back) doesn't need to care which backend wrote them.
    Endpoints (image 0 and -1) are never sent to CP2K — same convention as
    VASP, they keep their MACE-derived energy/forces.
    """
    cfg_dft_root = os.path.join(OUTPUT_ROOT, "dft_refinement", name)
    written_dirs = []
    for idx, atoms in enumerate(mace_images[1:-1], start=1):
        image_dir = os.path.join(cfg_dft_root, f"image_{idx:02d}")
        input_name = f"{name}_image_{idx:02d}"
        write_cp2k_sp(atoms, input_name, image_dir)
        written_dirs.append(image_dir)

    manifest = {
        "name": name,
        "dft_type": "cp2k",
        "cp2k_run_type": CP2K_RUN_TYPE,
        "n_images_total": len(mace_images),
        "n_moving_images": len(written_dirs),
        "moving_image_dirs": [os.path.basename(d) for d in written_dirs],
        "input_filename_pattern": f"{name}_image_XX.inp",
        "expected_output_pattern": f"{name}_image_XX.out",
    }
    with open(os.path.join(cfg_dft_root, "cp2k_inputs_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"    [✓] Wrote CP2K inputs for {len(written_dirs)} moving image(s) of '{name}' to "
          f"{cfg_dft_root}/image_XX/ — no CP2K run was launched.")
    return cfg_dft_root


def read_cp2k_outputs_from_disk(name: str, mace_images: list) -> list:
    """
    CP2K counterpart to read_vasp_outputs_from_disk(): scans
    {OUTPUT_ROOT}/dft_refinement/{name}/image_XX/{name}_image_XX.out for
    results copied back from the HPC, parses them with parse_cp2k_out(),
    and assembles refined_images in the same shape the rest of the
    pipeline expects. Endpoints keep their MACE values, same as VASP.
    A missing/unconverged/unparseable .out leaves that image at its MACE
    value and flags it, rather than failing the whole config.
    """
    cfg_dft_root = os.path.join(OUTPUT_ROOT, "dft_refinement", name)
    refined_images = [img.copy() for img in mace_images]
    _reattach_endpoint_calcs(refined_images, mace_images)

    n_expected = len(refined_images) - 2
    n_found = 0
    for idx in range(1, len(refined_images) - 1):
        image_dir = os.path.join(cfg_dft_root, f"image_{idx:02d}")
        out_path, pos_path = _find_cp2k_output_files(image_dir)

        if out_path is None:
            print(f"      [!] {name}/image_{idx:02d}: no .out file found in "
                  f"{image_dir}/ yet — keeping MACE value for this image.")
            continue

        e, f, stress_voigt, ok = parse_cp2k_out(out_path, refined_images[idx])
        if not ok or e is None:
            print(f"      [✗] {name}/image_{idx:02d}: could not parse energy from "
                  f"{Path(out_path).name} — keeping MACE value for this image.")
            continue

        if CP2K_RUN_TYPE == "geo_opt" and pos_path is not None:
            try:
                final_geom = read(pos_path, index=-1)
                if len(final_geom) == len(refined_images[idx]):
                    refined_images[idx].set_positions(final_geom.get_positions())
                else:
                    print(f"      [!] {name}/image_{idx:02d}: atom count mismatch in "
                          f"{Path(pos_path).name} — keeping pre-opt positions.")
            except Exception as err:
                print(f"      [!] {name}/image_{idx:02d}: could not read final geometry "
                      f"from {Path(pos_path).name} ({err}) — keeping pre-opt positions.")
        elif CP2K_RUN_TYPE == "geo_opt" and pos_path is None:
            print(f"      [~] {name}/image_{idx:02d}: geo_opt but no *-pos-1.xyz found "
                  f"alongside {Path(out_path).name} — positions are still pre-opt MACE geometry.")

        # Build SinglePointCalculator — forces optional (energy-only is fine
        # for energy-profile comparison; forces being None just means the
        # pathology/force-RMSE checks will skip this image gracefully)
        calc_kwargs = {"energy": e}
        if f is not None:
            calc_kwargs["forces"] = f
        if stress_voigt is not None:
            calc_kwargs["stress"] = stress_voigt
        refined_images[idx].calc = SinglePointCalculator(refined_images[idx], **calc_kwargs)
        n_found += 1

    status = "✓" if n_found == n_expected else "~"
    print(f"    [{status}] {name}: read {n_found}/{n_expected} moving-image CP2K output(s) from disk.")
    if n_found == 0:
        return []
    return refined_images


def write_dft_inputs(name: str, mace_images: list) -> str:
    """Dispatches to the VASP or CP2K input writer based on DFT_TYPE."""
    if DFT_TYPE == "cp2k":
        return write_cp2k_inputs(name, mace_images)
    return write_vasp_inputs(name, mace_images)


def read_dft_outputs_from_disk(name: str, mace_images: list) -> list:
    """Dispatches to the VASP or CP2K output reader based on DFT_TYPE."""
    if DFT_TYPE == "cp2k":
        return read_cp2k_outputs_from_disk(name, mace_images)
    return read_vasp_outputs_from_disk(name, mace_images)


def run_all_dft_refinements(neb_results: dict, mode: str = "run") -> dict:
    """
    Run the DFT polish once per config (not once per model pair).

    mode:
      "run"           — default; launches VASP via ASE's NEB+MDMin driver, as before.
      "write-inputs"  — writes POSCAR/INCAR/KPOINTS/POTCAR per moving image and
                         does NOT launch VASP or compute anything. Hand the
                         resulting dft_refinement/{name}/image_XX/ directories
                         to an HPC scheduler.
      "read-results"  — does not launch anything; scans
                         dft_refinement/{name}/image_XX/OUTCAR for results you've
                         copied back from the HPC, and folds them into the
                         normal comparison/leaderboard pipeline.
    """
    print("\n" + "="*70)
    titles = {
        "run":           "PART 5 — DFT PATHWAY REFINEMENT (ground truth)",
        "write-inputs":  "PART 5 — WRITING VASP INPUTS ONLY (no VASP launched)",
        "read-results":  "PART 5 — READING VASP RESULTS FROM DISK",
    }
    print(f"  {titles[mode]}")
    print("="*70)
    
    dft_results = {}
    for name, model_results in neb_results.items():
        ref = model_results.get(REFERENCE_MODEL_KEY, {})
        mace_images = ref.get("images")
        model_lbl   = ref.get("model_label", MODELS[REFERENCE_MODEL_KEY]["label"])

        if not mace_images:
            print(f"  [!] {name}: reference model '{REFERENCE_MODEL_KEY}' has no usable "
                  f"path — skipping DFT refinement for this config.")
            continue

        if mode == "write-inputs":
            try:
                write_dft_inputs(name, mace_images)
            except Exception as e:
                print(f"  [✗] Writing VASP inputs for {name} errored: {e}")
            continue   # nothing to compare/plot yet — VASP hasn't run

        if mode == "read-results":
            try:
                refined = read_dft_outputs_from_disk(name, mace_images)
            except Exception as e:
                print(f"  [✗] Reading VASP results for {name} errored: {e}")
                continue
        else:
            try:
                refined = run_dft_path_refinement(name, mace_images, model_lbl)
            except Exception as e:
                print(f"  [✗] DFT refinement for {name} errored: {e}")
                continue

        if not refined:
            continue

        if mode == "read-results":
            # Persist so a later plain --dft-only run finds this via
            # load_saved_dft_result() instead of needing OUTCARs rescanned.
            save_dft_result(name, refined)

        dft_ref = extract_dft_reference(name, refined)
        if dft_ref:
            dft_results[name] = dft_ref
            print(f"      • DFT forward barrier: {dft_ref['barrier_forward']:.4f} eV  "
                  f"(TS @ image {dft_ref['ts_image_index']})")
        plot_refined_dft_overlay(name, neb_results, refined)
        save_per_image_breakdown(name, neb_results, refined)

    return dft_results


# ==============================================================================
# PART 6 — LEADERBOARD: which MACE model performed best?
# ==============================================================================

def _consensus_disagreement(neb_results: dict) -> dict:
    """
    Fallback ranking signal when no DFT ground truth is available for a
    config: per-config cross-model median energy profile, then each
    model's RMSE from that median. Averaged over configs per model.
    """
    per_model_rmses = {k: [] for k in active_model_keys()}

    for name, model_results in neb_results.items():
        profiles = {}
        for k in active_model_keys():
            r = model_results.get(k, {})
            E = r.get("energies")
            if E and not r.get("error"):
                profiles[k] = np.array(E) - E[0]

        if len(profiles) < 2:
            continue

        n = min(len(p) for p in profiles.values())
        stacked = np.stack([p[:n] for p in profiles.values()])
        median_profile = np.median(stacked, axis=0)

        for k, p in profiles.items():
            rmse = float(np.sqrt(np.mean((p[:n] - median_profile) ** 2)))
            per_model_rmses[k].append(rmse)

    return {k: (float(np.mean(v)) if v else None) for k, v in per_model_rmses.items()}


def compute_pairwise_dft_comparison(neb_results: dict, dft_results: dict) -> dict:
    """
    All-pairs comparison across every model AND DFT (treated as one more
    "method") — complements compare_models() (which only does model-vs-model)
    by also putting DFT in the matrix, so you can see e.g. "is V4 actually
    closer to V5 than either is to DFT" at a glance, not just each model's
    aggregate distance from DFT (which is all the leaderboard shows).

    Uses the SAME --ignore-images mask as the leaderboard/DFT-reference
    step, so this stays consistent with whatever the leaderboard is scoring.

    Returns {"per_config": [...], "matrix": {method_a: {method_b: mean_rmse}}}.
    """
    print("\n" + "="*70)
    print("  PART 5b — PAIRWISE MODEL ⇄ DFT COMPARISON")
    print("="*70)

    keys = active_model_keys()
    methods = list(keys) + (["DFT"] if dft_results else [])
    if len(methods) < 2:
        print("  [!] Need at least 2 methods (models and/or DFT) to compare.")
        return {"per_config": [], "matrix": {}}

    def method_label(m):
        return "DFT" if m == "DFT" else MODELS[m]["label"]

    per_config_rows = []
    # pair_key -> list of (energy_rmse, |Δbarrier|) across configs, for the aggregate matrix
    pair_samples = {}

    for name, model_results in neb_results.items():
        profiles = {}   # method -> dict(energies_rel=..., barrier_forward=..., ts_image_index=...)
        for k in keys:
            r = model_results.get(k, {})
            E = r.get("energies")
            if not E or r.get("error"):
                continue
            E_masked = mask_ignored(name, E)
            profiles[k] = {
                "energies_rel": relative_to_first_finite(E_masked),
                "barrier_forward": r.get("barrier_forward"),
                "ts_image_index": r.get("ts_image_index"),
            }
        dft = dft_results.get(name)
        if dft:
            profiles["DFT"] = {
                "energies_rel": np.array(dft["energies_rel"], dtype=float),
                "barrier_forward": dft.get("barrier_forward"),
                "ts_image_index": dft.get("ts_image_index"),
            }

        available = [m for m in methods if m in profiles]
        if len(available) < 2:
            continue

        for m1, m2 in itertools.combinations(available, 2):
            p1, p2 = profiles[m1], profiles[m2]
            n = min(len(p1["energies_rel"]), len(p2["energies_rel"]))
            diff = p1["energies_rel"][:n] - p2["energies_rel"][:n]
            diff = diff[~np.isnan(diff)]
            energy_rmse = float(np.sqrt(np.mean(diff**2))) if len(diff) else None

            bf1, bf2 = p1["barrier_forward"], p2["barrier_forward"]
            barrier_diff = abs(bf2 - bf1) if (bf1 is not None and bf2 is not None) else None

            ts1, ts2 = p1["ts_image_index"], p2["ts_image_index"]
            ts_mismatch = (ts1 != ts2) if (ts1 is not None and ts2 is not None) else None

            per_config_rows.append({
                "name": name,
                "method_a": method_label(m1),
                "method_b": method_label(m2),
                "energy_rmse_eV": energy_rmse,
                "barrier_forward_a": bf1,
                "barrier_forward_b": bf2,
                "abs_barrier_diff_eV": barrier_diff,
                "ts_image_index_a": ts1,
                "ts_image_index_b": ts2,
                "ts_mismatch": ts_mismatch,
            })

            pair_key = tuple(sorted((m1, m2)))
            pair_samples.setdefault(pair_key, []).append((energy_rmse, barrier_diff))

    # ── Aggregate into a symmetric matrix (mean energy RMSE per pair) ───────
    matrix = {method_label(m): {method_label(m): None for m in methods} for m in methods}
    for (m1, m2), samples in pair_samples.items():
        rmses = [s[0] for s in samples if s[0] is not None]
        mean_rmse = float(np.mean(rmses)) if rmses else None
        matrix[method_label(m1)][method_label(m2)] = mean_rmse
        matrix[method_label(m2)][method_label(m1)] = mean_rmse

    # ── Print matrix ─────────────────────────────────────────────────────────
    labels = [method_label(m) for m in methods]
    col_w = max(14, max(len(l) for l in labels) + 2)
    print(f"\n  Mean energy RMSE across all configs (eV) — lower = more agreement")
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print(" " * 2 + header)
    for l1 in labels:
        row = f"  {l1:<{col_w-2}}"
        for l2 in labels:
            v = matrix[l1][l2]
            row += f"{'—' if l1 == l2 else (f'{v:.4f}' if v is not None else 'N/A'):>{col_w}}"
        print(row)

    if IGNORE_IMAGES:
        print(f"\n  (images excluded per --ignore-images: {IGNORE_IMAGES})")

    return {"per_config": per_config_rows, "matrix": matrix}


def save_pairwise_dft_comparison(result: dict):
    report_dir = os.path.join(OUTPUT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)

    json_path = os.path.join(report_dir, "pairwise_model_dft_comparison.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    csv_path = os.path.join(report_dir, "pairwise_model_dft_comparison.csv")
    rows = result.get("per_config", [])
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    matrix_csv_path = os.path.join(report_dir, "pairwise_model_dft_matrix.csv")
    matrix = result.get("matrix", {})
    if matrix:
        labels = list(matrix.keys())
        with open(matrix_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + labels)
            for l1 in labels:
                writer.writerow([l1] + [matrix[l1].get(l2) for l2 in labels])

    print(f"\n[✓] Pairwise model⇄DFT comparison saved: {json_path}")
    print(f"[✓] Pairwise model⇄DFT comparison saved: {csv_path}")
    print(f"[✓] Pairwise model⇄DFT matrix saved: {matrix_csv_path}")


def compute_leaderboard(neb_results: dict, all_pathologies: dict, dft_results: dict) -> list:
    """
    Build a ranked leaderboard of every model.

    Primary score (lower = better):
      - if DFT ground truth exists for >=1 config: mean absolute forward-
        barrier error vs DFT (eV), averaged over configs that have DFT data
      - otherwise: mean RMSE from the cross-model median energy profile
        (a "how much does this model disagree with the consensus" proxy)

    Also reports: convergence rate, mean wall time, total pathology flags,
    so a model can win on accuracy but still be flagged as expensive or
    unstable.
    """
    print("\n" + "="*70)
    print("  PART 6 — MODEL LEADERBOARD")
    print("="*70)

    keys = active_model_keys()
    have_dft = len(dft_results) > 0

    consensus = _consensus_disagreement(neb_results) if not have_dft else {}

    rows = []
    for k in keys:
        label = MODELS[k]["label"]
        n_configs = 0
        n_converged = 0
        wall_times = []
        n_pathology_flags = 0
        barrier_errors_vs_dft = []
        energy_rmse_vs_dft = []

        for name, model_results in neb_results.items():
            r = model_results.get(k)
            if r is None:
                continue
            if r.get("error") or not r.get("energies"):
                continue

            n_configs += 1
            if r.get("converged"):
                n_converged += 1
            if r.get("elapsed_s") is not None:
                wall_times.append(r["elapsed_s"])

            flags = all_pathologies.get(name, {}).get(k, [])
            n_pathology_flags += len(flags)

            dft = dft_results.get(name)
            if dft and r.get("barrier_forward") is not None:
                barrier_errors_vs_dft.append(abs(r["barrier_forward"] - dft["barrier_forward"]))

                E_model = mask_ignored(name, r["energies"])
                E_model_rel = relative_to_first_finite(E_model)
                n = min(len(E_model_rel), len(dft["energies_rel"]))
                diff = E_model_rel[:n] - np.array(dft["energies_rel"][:n])
                diff = diff[~np.isnan(diff)]
                if len(diff):
                    rmse = float(np.sqrt(np.mean(diff ** 2)))
                    energy_rmse_vs_dft.append(rmse)

        if n_configs == 0:
            continue

        row = {
            "model_key":            k,
            "label":                label,
            "n_configs_run":        n_configs,
            "convergence_rate":     n_converged / n_configs,
            "mean_wall_time_s":     float(np.mean(wall_times)) if wall_times else None,
            "total_pathology_flags": n_pathology_flags,
            "mean_abs_barrier_error_vs_dft_eV": (
                float(np.mean(barrier_errors_vs_dft)) if barrier_errors_vs_dft else None
            ),
            "mean_energy_rmse_vs_dft_eV": (
                float(np.mean(energy_rmse_vs_dft)) if energy_rmse_vs_dft else None
            ),
            "n_configs_with_dft":   len(barrier_errors_vs_dft),
            "consensus_rmse_eV":    consensus.get(k),
        }
        rows.append(row)

    def sort_key(row):
        if row["mean_abs_barrier_error_vs_dft_eV"] is not None:
            primary = row["mean_abs_barrier_error_vs_dft_eV"]
        elif row["consensus_rmse_eV"] is not None:
            primary = row["consensus_rmse_eV"]
        else:
            primary = float("inf")
        # tie-break: prefer higher convergence rate, fewer pathology flags
        return (primary, -row["convergence_rate"], row["total_pathology_flags"])

    rows.sort(key=sort_key)

    print(f"\n  Ranking basis: "
          f"{'mean |barrier error| vs DFT ground truth' if have_dft else 'agreement with cross-model consensus (no DFT data available)'}\n")

    header = f"  {'Rank':<5}{'Model':<16}{'Configs':>8}{'Converged':>11}{'Mean t (s)':>12}{'Pathology':>11}"
    if have_dft:
        header += f"{'|ΔBarrier| vs DFT':>20}{'E_RMSE vs DFT':>16}"
    else:
        header += f"{'Consensus RMSE':>16}"
    print(header)
    print("  " + "─"*len(header.strip()))

    for i, row in enumerate(rows, start=1):
        wt = f"{row['mean_wall_time_s']:.1f}" if row['mean_wall_time_s'] is not None else "N/A"
        conv = f"{row['convergence_rate']*100:.0f}%"
        line = (f"  {i:<5}{row['label']:<16}{row['n_configs_run']:>8}{conv:>11}{wt:>12}"
                f"{row['total_pathology_flags']:>11}")
        if have_dft:
            be = row["mean_abs_barrier_error_vs_dft_eV"]
            er = row["mean_energy_rmse_vs_dft_eV"]
            bes = f"{be:.4f}" if be is not None else "N/A (no DFT)"
            ers = f"{er:.4f}" if er is not None else "N/A"
            line += f"{bes:>20}{ers:>16}"
        else:
            cr = row["consensus_rmse_eV"]
            line += f"{(f'{cr:.4f}' if cr is not None else 'N/A'):>16}"
        print(line)

    print()
    if rows:
        winner = rows[0]
        print(f"  🏆  Best-performing model this run: {winner['label']}")
        if not have_dft:
            print("      (No DFT data was available — this ranking is based on consensus "
                  "agreement only. Re-run with RUN_DFT_REFINEMENT=True for a ground-truth result.)")
    print()

    return rows


def save_per_image_breakdown(name: str, neb_results: dict, refined_images: list):
    """
    Write a CSV for this config with one row per NEB image and one column
    per model (+ 'DFT' if available), all in ABSOLUTE energy units (eV) —
    i.e. raw get_potential_energy() values, not relative-to-endpoint. An
    'ignored' column flags any image excluded from plots/metrics via
    --ignore-images, but the energy value itself is still shown here.
    """
    report_dir = os.path.join(OUTPUT_ROOT, "reports", "per_image_breakdown")
    os.makedirs(report_dir, exist_ok=True)

    columns = {}
    n_images = 0
    for k in active_model_keys():
        r = neb_results.get(name, {}).get(k, {})
        E = r.get("energies")
        if E:
            columns[MODELS[k]["label"]] = list(E)
            n_images = max(n_images, len(E))

    if refined_images:
        try:
            dft_e = [img.get_potential_energy() for img in refined_images]
            columns["DFT"] = dft_e
            n_images = max(n_images, len(dft_e))
        except Exception as e:
            print(f"      [~] Could not include DFT column in per-image breakdown for {name}: {e}")

    if not columns or n_images == 0:
        return None

    ignored = get_ignored_images(name)
    csv_path = os.path.join(report_dir, f"{name}_per_image.csv")
    fieldnames = ["image_index", "ignored"] + list(columns.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n_images):
            row = {"image_index": idx, "ignored": idx in ignored}
            for col, vals in columns.items():
                row[col] = vals[idx] if idx < len(vals) else None
            writer.writerow(row)

    print(f"    [✓] Per-image breakdown saved: {csv_path}")
    return csv_path


def save_leaderboard(rows: list):
    report_dir = os.path.join(OUTPUT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)

    json_path = os.path.join(report_dir, "model_leaderboard.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    csv_path = os.path.join(report_dir, "model_leaderboard.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"[✓] Leaderboard saved: {json_path}")
    print(f"[✓] Leaderboard saved: {csv_path}")


# ==============================================================================
# OUTPUT & REPORTING
# ==============================================================================

def save_json_report(
    validation_results: list,
    comparisons: list,
    all_pathologies: dict,
    dft_results: dict,
):
    """Save all results to a machine-readable JSON report."""
    report_dir = os.path.join(OUTPUT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)

    # Pathologies are nested dicts of lists — already JSON-serialisable
    report = {
        "run_timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_model":    REFERENCE_MODEL_KEY,
        "models":             {k: {"label": v["label"], "path": v["path"]}
                                for k, v in MODELS.items()},
        "validation":         validation_results,
        "pairwise_comparisons": [
            {k: v for k, v in c.items()
             if k not in ("energies_a", "energies_b", "max_forces_a", "max_forces_b")}
            for c in comparisons
        ],
        "pathologies":        all_pathologies,
        "dft_reference": {
            name: {k: v for k, v in d.items() if k != "energies" and k != "energies_rel"}
            for name, d in dft_results.items()
        },
    }

    path = os.path.join(report_dir, "neb_comparison_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[✓] JSON report saved: {path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    global RUN_DFT_REFINEMENT, FORCE_DFT
    global OUTPUT_ROOT, MODELS, REFERENCE_MODEL_KEY, FORCE_DFT
    global IGNORE_IMAGES, DFT_TYPE, CP2K_RUN_TYPE, CP2K_GEO_OPT_MAX_STEPS

    parser = argparse.ArgumentParser(
        description="MACE multi-model NEB comparison for Pt dissolution pathways"
    )
    parser.add_argument("--csv",             type=str, default=PATH_CSV,
                        help="Path to config CSV (Name, initial, final)")
    parser.add_argument("--validate-only",   action="store_true",
                        help="Only run structure validation, skip NEB")
    parser.add_argument("--no-neb",          action="store_true",
                        help="Skip endpoint relaxation and NEB optimisation; instead reload "
                             "each model/config's previously-saved NEB result from disk "
                             "(OUTPUT_ROOT/{model}/{config}/{config}_neb.extxyz) and resume "
                             "straight into comparison/pathology/DFT-refinement/leaderboard. "
                             "Use this to re-run only the downstream steps (e.g. after fixing "
                             "a VASP config) without recomputing MACE NEB.")
    parser.add_argument("--dft-only",        action="store_true",
                        help="Implies --no-neb (MACE NEB results are reloaded from disk, no "
                             "MACE models are even loaded onto the GPU) and runs straight "
                             "through to DFT refinement + leaderboard. Configs whose DFT "
                             "polish already finished are loaded from "
                             "OUTPUT_ROOT/dft_refinement/{config}/{config}_dft_refined.extxyz "
                             "instead of resubmitting VASP — see --force-dft to override.")
    parser.add_argument("--force-dft",       action="store_true",
                        help="Re-run VASP NEB polish even for configs that already have a "
                             "cached {config}_dft_refined.extxyz on disk.")
    parser.add_argument("--dft-type",        type=str, default=DFT_TYPE,
                        choices=["vasp", "cp2k"],
                        help="DFT backend for the ground-truth refinement step. 'vasp' "
                             "(default) supports both the live ASE-driven NEB polish ('run' "
                             "mode) and write-inputs/read-results. 'cp2k' supports "
                             "write-inputs/read-results only (matches HPC submission "
                             "workflows) — use --write-dft-inputs / --read-dft-results.")
    parser.add_argument("--cp2k-run-type",   type=str, default=CP2K_RUN_TYPE,
                        choices=["sp", "geo_opt"],
                        help="'sp' (default): single-point ENERGY_FORCE, geometry untouched "
                             "— analogous to VASP's ibrion=-1/nsw=0. 'geo_opt': small "
                             "CP2K-internal relax (see --cp2k-geo-opt-steps) before reporting "
                             "final E/F/positions — analogous to VASP's ibrion=2. Only used "
                             "when --dft-type cp2k.")
    parser.add_argument("--cp2k-geo-opt-steps", type=int, default=CP2K_GEO_OPT_MAX_STEPS,
                        help=f"MAX_ITER for CP2K's &GEO_OPT when --cp2k-run-type geo_opt "
                             f"(default: {CP2K_GEO_OPT_MAX_STEPS}).")
    parser.add_argument("--write-vasp-inputs", "--write-dft-inputs",
                        dest="write_dft_inputs", action="store_true",
                        help="Write DFT input files for every moving image of the "
                             "DFT-refinement pathway WITHOUT launching a DFT job — VASP "
                             "(POSCAR/INCAR/KPOINTS/POTCAR) or CP2K (.inp), depending on "
                             "--dft-type. Implies --no-neb. Use this to prepare "
                             "{output-dir}/dft_refinement/{config}/image_XX/ directories for "
                             "submission to an HPC scheduler; copy the resulting OUTCAR (VASP) "
                             "or .out (CP2K) files back into the same folders, then re-run "
                             "with --read-dft-results.")
    parser.add_argument("--read-vasp-results", "--read-dft-results",
                        dest="read_dft_results", action="store_true",
                        help="Skip launching DFT entirely; scan "
                             "{output-dir}/dft_refinement/{config}/image_XX/ for results "
                             "already copied back from an HPC run (OUTCAR for VASP, "
                             "{config}_image_XX.out for CP2K, per --dft-type) and fold them "
                             "into the normal comparison/leaderboard pipeline (same as a "
                             "completed --dft-only run). Implies --no-neb.")
    parser.add_argument('--no-dft', dest='run_dft', action='store_false', 
                        help="Skip the DFT / VASP pathway refinement step entirely")
    parser.add_argument("--endpoint-relax-mode", type=str, default=ENDPOINT_RELAX_MODE,
                        choices=["per-model", "shared"],
                        help="'shared' (default): each model relaxes its own copy of the "
                             "endpoints before its NEB. 'shared': legacy behaviour — only "
                             "REFERENCE_MODEL_KEY relaxes the endpoints once and every model "
                             "reuses that single geometry. Ignored if --no-neb is set.")
    parser.add_argument("--output-dir",      type=str, default=OUTPUT_ROOT,
                        help="Root output directory")
    parser.add_argument("--reference-model", type=str, default=REFERENCE_MODEL_KEY,
                        help="Model key to use for endpoint pre-relaxation and as the "
                             "DFT-polish initial guess")
    parser.add_argument("--only-models",     type=str, default=None,
                        help="Comma-separated subset of model keys to run "
                             "(default: all models in MODELS)")
    parser.add_argument("--skip-models",     type=str, default=None,
                        help="Comma-separated model keys to exclude from this run")
    parser.add_argument("--ignore-images",   type=str, default=None,
                        help="Exclude specific NEB image indices from plots/RMSE/barrier "
                             "scoring (data is still shown in the per-image CSV breakdown, "
                             "just flagged). Format: '0,13' to ignore those indices in every "
                             "config, or 'Close0.75PtO2:0,13;Close0.5Pt:5' for per-config "
                             "rules (';'-separated, mix freely with the global form).")
    args = parser.parse_args()

    OUTPUT_ROOT = args.output_dir
    REFERENCE_MODEL_KEY = args.reference_model
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    DFT_TYPE = args.dft_type
    CP2K_RUN_TYPE = args.cp2k_run_type
    CP2K_GEO_OPT_MAX_STEPS = args.cp2k_geo_opt_steps
    print(f"[i] DFT backend: {DFT_TYPE.upper()}"
          f"{f' (CP2K run type: {CP2K_RUN_TYPE}, max_iter={CP2K_GEO_OPT_MAX_STEPS})' if DFT_TYPE == 'cp2k' else ''}")

    if args.dft_only:
        args.no_neb = True
        print("[i] --dft-only set: implies --no-neb (resuming MACE NEB results from disk).")

    if args.write_dft_inputs or args.read_dft_results:
        args.no_neb = True
        flag_name = "write-dft-inputs" if args.write_dft_inputs else "read-dft-results"
        print(f"[i] --{flag_name} set: implies --no-neb (resuming MACE NEB results from disk; "
              f"no MACE compute needed to {'write DFT inputs' if args.write_dft_inputs else 'read DFT results'}).")

    FORCE_DFT = args.force_dft

    if args.ignore_images:
        IGNORE_IMAGES = parse_ignore_images(args.ignore_images)
        print(f"[i] Ignoring images per --ignore-images: {IGNORE_IMAGES}")

    if args.only_models:
        keep = set(s.strip() for s in args.only_models.split(","))
        MODELS = {k: v for k, v in MODELS.items() if k in keep}
    if args.skip_models:
        drop = set(s.strip() for s in args.skip_models.split(","))
        MODELS = {k: v for k, v in MODELS.items() if k not in drop}

    if not MODELS:
        raise RuntimeError("No models left to run after applying --only-models/--skip-models.")

    if REFERENCE_MODEL_KEY not in MODELS:
        REFERENCE_MODEL_KEY = next(iter(MODELS.keys()))
        print(f"[!] Requested reference model not in the active model set. "
              f"Using '{REFERENCE_MODEL_KEY}' instead.")

    print("\n" + "="*70)
    print("  MACE MULTI-MODEL NEB COMPARISON")
    print("="*70)
    print(f"  Models:     {', '.join(m['label'] for m in MODELS.values())}")
    print(f"  Reference:  {MODELS[REFERENCE_MODEL_KEY]['label']}")
    print(f"  CSV:        {args.csv}")
    print(f"  Output dir: {OUTPUT_ROOT}")
    print(f"  Endpoint relax mode: {'N/A (--no-neb, resuming from disk)' if args.no_neb else args.endpoint_relax_mode}")
    print(f"  Started:    {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    t_run_start = time.perf_counter()

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        configs = [
            {"name": row["Name"], "initial": row["initial"], "final": row["final"]}
            for row in reader
        ]
    print(f"[✓] Loaded {len(configs)} configs from CSV.\n")

    with timed("validation"):
        validation_results = run_all_validations(configs)

    if args.validate_only:
        save_json_report(validation_results, [], {}, {})
        TIMINGS["total_runtime"] = time.perf_counter() - t_run_start
        print_timing_summary()
        save_timing_summary()
        print("[✓] Validation-only run complete.")
        return

    with timed("model_loading"):
        if args.no_neb:
            # Resuming from disk (--no-neb / --dft-only): no MACE compute is
            # needed for the resume path, so skip loading models onto the
            # GPU entirely — we only need the set of model keys to know
            # which {model}/{config}/{config}_neb.extxyz files to look for.
            print("[i] --no-neb: skipping MACE model loading (not needed to resume from disk).")
            calcs = {k: None for k in MODELS.keys()}
        else:
            calcs = load_models()

    # NOTE: run_all_nebs() is ALWAYS called now, --no-neb included. Internally
    # it branches: --no-neb reloads saved results from disk (no relaxation,
    # no MACE NEB compute) so the run can resume straight into comparison/
    # pathology/DFT-refinement/leaderboard. Previously this branch was never
    # reached because of an `if not args.no_neb` guard here, which meant
    # --no-neb silently did nothing but validate and exit — fixed.
    neb_results = run_all_nebs(
        configs, calcs, validation_results,
        no_neb=args.no_neb,
        endpoint_relax_mode=args.endpoint_relax_mode,
    )

    if not neb_results or not any(neb_results.values()):
        print("[!] No NEB results to analyse. Exiting.")
        if args.no_neb:
            print("    (--no-neb was set but no saved {config}_neb.extxyz files were found "
                  f"under {OUTPUT_ROOT}/{{model}}/{{config}}/ — nothing to resume from. "
                  "Run without --no-neb first.)")
        save_json_report(validation_results, [], {}, {})
        TIMINGS["total_runtime"] = time.perf_counter() - t_run_start
        print_timing_summary()
        save_timing_summary()
        return

    with timed("comparison"):
        comparisons = compare_models(neb_results)
        plot_comparison(comparisons, neb_results)

    with timed("pathology_detection"):
        all_pathologies = detect_pathologies(neb_results)
        plot_pathology_summary(all_pathologies, neb_results)

    if not args.run_dft and not (args.write_dft_inputs or args.read_dft_results):
        RUN_DFT_REFINEMENT = False
        print(f"[!] --no-dft passed: {DFT_TYPE.upper()} calculations will be skipped for this run.")

    if args.force_dft:
        FORCE_DFT = True

    if args.write_dft_inputs:
        with timed("dft_refinement"):
            run_all_dft_refinements(neb_results, mode="write-inputs")
        # Nothing to compare yet (DFT hasn't run), so stop before the
        # leaderboard rather than writing one out with empty DFT data.
        save_json_report(validation_results, comparisons, all_pathologies, {})
        TIMINGS["total_runtime"] = time.perf_counter() - t_run_start
        print_timing_summary()
        save_timing_summary()
        out_label = "OUTCAR" if DFT_TYPE == "vasp" else "{config}_image_XX.out"
        print(f"\n[✓] {DFT_TYPE.upper()} inputs written under {OUTPUT_ROOT}/dft_refinement/{{config}}/image_XX/.")
        print(f"    Copy those directories to your HPC, run {DFT_TYPE.upper()}, copy the resulting "
              f"{out_label} files back into the same image_XX/ folders, then re-run this script "
              f"with --read-dft-results to pull the results back in.")
        return

    dft_results = {}
    if args.read_dft_results:
        with timed("dft_refinement"):
            dft_results = run_all_dft_refinements(neb_results, mode="read-results")
    elif RUN_DFT_REFINEMENT:
        try:
            with timed("dft_refinement"):
                dft_results = run_all_dft_refinements(neb_results, mode="run")
        except Exception as e:
            print(f"\n[✗] DFT refinement process encountered an error: {e}")
            print("    Skipping DFT refinement and proceeding to leaderboard/reporting.\n")

    with timed("pairwise_dft_comparison"):
        pairwise_dft_result = compute_pairwise_dft_comparison(neb_results, dft_results)
        save_pairwise_dft_comparison(pairwise_dft_result)

    with timed("leaderboard"):
        leaderboard_rows = compute_leaderboard(neb_results, all_pathologies, dft_results)
        save_leaderboard(leaderboard_rows)

    # Configs that never went through run_all_dft_refinements (DFT skipped
    # entirely, or that config's DFT step failed) still get a per-image
    # breakdown CSV, just without a DFT column.
    for name in neb_results.keys():
        if name not in dft_results:
            save_per_image_breakdown(name, neb_results, [])

    save_json_report(validation_results, comparisons, all_pathologies, dft_results)

    TIMINGS["total_runtime"] = time.perf_counter() - t_run_start
    print_timing_summary()
    save_timing_summary()

    print(f"[✓] All done. Results in: {OUTPUT_ROOT}/")
    for k in MODELS.keys():
        print(f"    ├── {k}/                ← NEB files for {MODELS[k]['label']}")
    print(f"    ├── comparison_plots/        ← overlay energy profiles + barrier bar charts")
    print(f"    ├── pathology_plots/         ← flagged image plots")
    print(f"    ├── dft_refinement/          ← VASP NEB polish per config")
    print(f"    └── reports/")
    print(f"          ├── neb_comparison_report.json")
    print(f"          ├── model_leaderboard.json   ← which model performed best")
    print(f"          ├── model_leaderboard.csv")
    print(f"          ├── pairwise_model_dft_comparison.csv  ← per-config method-vs-method energy RMSE/Δbarrier")
    print(f"          ├── pairwise_model_dft_matrix.csv      ← aggregated mean RMSE matrix (all methods incl. DFT)")
    print(f"          ├── per_image_breakdown/     ← absolute-energy CSV per config, per image")
    print(f"          └── timings.json             ← wall-clock time per phase")


if __name__ == "__main__":
    main()