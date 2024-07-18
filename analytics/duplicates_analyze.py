#!/usr/bin/env python3
from typing import Dict, List
from subprocess import check_output
import shutil
import os
import sys


def get_defined_symbols(fname: str, verbose: bool = False) -> Dict[str, int]:
    if verbose:
        print(f"Processing {fname}...", end='', flush=True)
    if sys.platform == 'darwin':
        nm_path = shutil.which('nm')
        if nm_path is None:
            raise FileNotFoundError("The 'nm' command is not found in the system PATH.")
        lines = check_output([nm_path, '--defined-only', '-n', fname]).decode('ascii').split("\n")[:-1]
        rc = {}
        for idx, line in enumerate(lines):
            addr, stype, name = line.split(' ')
            size = 4 if idx + 1 == len(lines) else (int(lines[idx + 1].split(' ')[0], 16) - int(addr, 16))
            rc[name] = size
    else:
        nm_path = shutil.which('nm')
        if nm_path is None:
            raise FileNotFoundError("The 'nm' command is not found in the system PATH.")
        if not os.path.isfile(fname):
            raise ValueError(f"The file {fname} does not exist or is not a valid file.")
        lines = check_output([nm_path, '--print-size', '--defined-only', fname]).decode('ascii').split('\n')
        rc = {e[3]: int(e[1], 16) for e in [line.split() for line in lines] if len(e) == 4}
    if verbose:
        print("done")
    return rc


def get_deps(fname: str) -> List[str]:
    if sys.platform == 'darwin':
        rc = []
        otool_path = shutil.which('otool')
        if otool_path is None:
            raise FileNotFoundError("The 'otool' command is not found in the system PATH.")
        if not os.path.isfile(fname):
            raise ValueError(f"The file {fname} does not exist or is not a valid file.")
        lines = check_output([otool_path, '-l', fname]).decode('ascii').split("\n")[1:-1]
        for idx, line in enumerate(lines):
            if line.strip() != 'cmd LC_LOAD_DYLIB':
                continue
            path = lines[idx + 2].strip()
            if not path.startswith('name'):
                raise ValueError(f"Expected path to start with 'name', but got: {path}")
            rc.append(os.path.basename(path.split(' ')[1]))
        return rc
    readelf_path = shutil.which('readelf')
    if readelf_path is None:
        raise FileNotFoundError("The 'readelf' command is not found in the system PATH.")
    if readelf_path is None:
        raise FileNotFoundError("The 'readelf' command is not found in the system PATH.")
    if not os.path.isfile(fname):
        raise ValueError(f"The file {fname} does not exist or is not a valid file.")
    lines = check_output([readelf_path, '--dynamic', fname]).decode('ascii').split('\n')
    return [line.split('[')[1][:-1] for line in lines if '(NEEDED)' in line]


def humansize(size):
    if size < 1024:
        return f"{size} bytes"
    if size < 1024**2:
        return f"{int(size/1024)} Kb"
    if size < 1024**3:
        return f"{size/(1024.0**2):.2f} Mb"
    return f"{size/(1024.0**3):.2f} Gb"


def print_sizes(libname, depth: int = 2) -> None:
    libs = [libname]
    depth = 2
    symbols = {os.path.basename(libname): get_defined_symbols(libname, verbose=True)}
    for _ in range(depth):
        for lib in libs:
            dirname = os.path.dirname(lib)
            for dep in get_deps(lib):
                path = os.path.join(dirname, dep)
                if not os.path.exists(path):
                    continue
                if path not in libs:
                    libs.append(path)
                    symbols[dep] = get_defined_symbols(path, verbose=True)

    for lib in libs:
        lib_symbols = symbols[os.path.basename(lib)]
        lib_keys = set(lib_symbols.keys())
        rc = f"{lib} symbols size {humansize(sum(lib_symbols.values()))}"
        for dep in get_deps(lib):
            if dep not in symbols:
                continue
            dep_overlap = lib_keys.intersection(set(symbols[dep].keys()))
            overlap_size = sum(lib_symbols[k] for k in dep_overlap)
            if overlap_size > 0:
                rc += f" {dep} overlap is {humansize(overlap_size)}"
        print(rc)


def print_symbols_overlap(libname1: str, libname2: str) -> None:
    sym1 = get_defined_symbols(libname1, verbose=True)
    sym2 = get_defined_symbols(libname2, verbose=True)
    sym1_size = sum(sym1.values())
    sym2_size = sum(sym2.values())
    sym_overlap = set(sym1.keys()).intersection(set(sym2.keys()))
    overlap_size = sum(sym1[s] for s in sym_overlap)
    if overlap_size == 0:
        print(f"{libname1} symbols size {humansize(sym1_size)} does not overlap with {libname2}")
        return
    print(f"{libname1} symbols size {humansize(sym1_size)} overlap {humansize(overlap_size)} ({100.0 * overlap_size/sym1_size :.2f}%)")
    for sym in sym_overlap:
        print(sym)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        print_symbols_overlap(sys.argv[1], sys.argv[2])
    else:
        print_sizes(sys.argv[1] if len(sys.argv) > 1 else "lib/libtorch_cuda.so")
