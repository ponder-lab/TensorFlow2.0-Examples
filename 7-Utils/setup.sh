#!/bin/bash
# setup.sh — materialize the MNIST directory `multi_gpu_train.py` reads.
#
# The driver calls `flow_from_directory` on `./mnist/train` and `./mnist/test` with
# `class_mode='categorical'`, which requires a `<split>/<class>/<image>` layout. This repository
# already vendors those digits, under `../4-Object_Detection/YOLOV3/yymnist/mnist`, but in a
# shape that loader cannot read, for two independent reasons:
#
#   - The files are FLAT, with the label carried in the filename (`0_00007.pgm`). Given a
#     directory with no class subdirectories, `flow_from_directory` finds zero classes.
#   - Every file is `.pgm`, and Keras accepts png, jpg, jpeg, bmp, ppm, tif and tiff. Note that
#     `ppm` is accepted and `pgm` is not, a near-miss that reads as supported at a glance.
#
# Either alone is enough for the driver to report "Found 0 images belonging to 0 classes" and
# then train on nothing, rather than failing. That is the point of this script: an earlier
# attempt to symlink the vendored directory into place (`8a121f2`, reverted the same day by
# `549356b`) could not have worked, for those reasons.
#
# Nothing is downloaded and nothing is synthesized. The digits are read from the vendored source
# and rewritten into the layout the program already declares. `yymnist/make_data.py` is not the
# tool for this: it produces detection data with a labels file, and `mnist/` is its raw input.
#
# TWO PROPERTIES THAT ARE REQUIREMENTS RATHER THAN PREFERENCES, because a harness may run this
# once per repetition and compare the runs against each other:
#
#   DETERMINISTIC. No shuffle, no sampling, no re-drawn split. The vendored `train` and `test`
#   directories already carry the split and it is preserved rather than reinvented. Traversal is
#   sorted, so the output does not depend on the filesystem's directory ordering. If the output
#   varied between runs, the runs would differ by their data rather than by what is under test.
#
#   IDEMPOTENT. A second invocation must be a no-op. Guarded on a completion marker, following
#   the sibling `4-Object_Detection/YOLOV3/setup.sh`, which guards the same way.
set -eu

PYTHON="${PYTHON:-/usr/local/bin/python3.10}"
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/../4-Object_Detection/YOLOV3/yymnist/mnist"
dst="$here/mnist"

# The marker records COMPLETION rather than mere existence: a run interrupted part-way leaves the
# directory populated but short, and a bare `[ -d mnist ]` guard would then skip the repair.
if [[ -f "$dst/.materialized" ]]; then
    exit 0
fi

# The digits live in the nested `yymnist` submodule, so a clone that did not recurse has the
# directory but not its contents. Say which submodule rather than only which path is missing.
if [[ ! -d "$src" ]]; then
    echo "setup.sh: no vendored digits at $src" >&2
    echo "          initialize the submodule first:" >&2
    echo "          git submodule update --init 4-Object_Detection/YOLOV3/yymnist" >&2
    exit 1
fi

"$PYTHON" - "$src" "$dst" <<'PY'
import os, sys
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
total = 0
for split in ("train", "test"):
    srcdir = os.path.join(src, split)
    if not os.path.isdir(srcdir):
        sys.exit("setup.sh: missing split %s" % srcdir)
    for name in sorted(os.listdir(srcdir)):
        if not name.endswith(".pgm"):
            continue
        digit = name.split("_", 1)[0]
        if not digit.isdigit():
            sys.exit("setup.sh: cannot read a label from %s" % name)
        outdir = os.path.join(dst, split, digit)
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, name[:-4] + ".png")
        if not os.path.exists(out):
            Image.open(os.path.join(srcdir, name)).save(out, "PNG")
        total += 1
print("materialized %d images under %s" % (total, dst))
PY

touch "$dst/.materialized"
