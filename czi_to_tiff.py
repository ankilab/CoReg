#!/usr/bin/env python3
"""Convert a .czi file into a multi-page TIFF stack.

Usage:
  python czi_to_tiff.py input.czi [output.tif]

This script uses `czifile` to read the CZI and `tifffile` to write the TIFF stack.
It collapses all non-spatial axes into TIFF pages (keeps last two dims as Y,X).
"""
import argparse
import os
from czifile import CziFile
import tifffile


def czi_to_tiff_stack(infile: str, outfile: str, compression: str | None = "zlib") -> None:
    with CziFile(infile) as czi:
        arr = czi.asarray()

    if arr.ndim < 2:
        raise ValueError(f"Unexpected array shape: {arr.shape}")

    # Collapse all leading axes (e.g., T, C, Z) into a single page axis and keep Y, X
    yx_shape = arr.shape[-2:]
    pages = arr.reshape(-1, yx_shape[0], yx_shape[1])

    tifffile.imwrite(outfile, pages, photometric="minisblack", compression=compression)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert CZI to multi-page TIFF stack.")
    p.add_argument("infile", help="Input .czi file")
    p.add_argument("outfile", nargs="?", help="Output .tif file (optional)")
    p.add_argument("--no-compress", dest="compress", action="store_const", const=None, default="zlib",
                   help="Disable compression in output TIFF")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    infile = args.infile
    outfile = args.outfile or os.path.splitext(infile)[0] + ".tif"

    print(f"Reading: {infile}")
    czi_to_tiff_stack(infile, outfile, compression=args.compress)
    print(f"Wrote: {outfile}")


if __name__ == "__main__":
    main()
