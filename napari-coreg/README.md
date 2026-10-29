# napari-coreg

A napari widget around CoReg's CoTracker-based motion-correction pipeline.

This is a **separate, independently installable package** from the
standalone CoReg GUI (`src/CoReg/gui.py`). It shares none of that GUI's
code or Qt bindings — it only imports the headless, Qt-free pieces of the
main package (`utils.cotracker_pipeline`, `utils.presets`). That separation
is intentional:

- The standalone GUI pins `PyQt6`. napari ships its own Qt abstraction
  (`qtpy`, backed by whichever of PyQt5/PySide2/PyQt6/PySide6 is
  installed). Keeping this plugin in its own package means installing one
  never drags in the other's Qt stack.
- This plugin can be released/versioned on its own schedule (e.g. to the
  napari hub) without being coupled to the GUI's release cycle.

## Install

`coreg` (this repo's `src/CoReg`) isn't published to PyPI, so install both
packages yourself, in either order:

```bash
pip install -e /path/to/registration_gui   # provides utils.cotracker_pipeline / utils.presets
pip install -e /path/to/registration_gui/napari-coreg
```

`torch`'s own package metadata declares `triton` as a dependency; pip
re-installs it on *any* subsequent `pip install` in this env (even one
unrelated to torch), which reintroduces a GPU/OpenGL-related segfault --
see `requirements.txt`'s comment on it. After any `pip install` here, run:

```bash
pip uninstall -y triton
```

Then launch napari and open the plugin from
`Plugins > CoReg > CoReg Motion Correction`.

## Usage

1. Open or drag in a video stack as an Image layer (`(T, H, W)` or
   `(T, H, W, 3)`).
2. Pick that layer in the widget, choose a seed-point preset and its
   parameters -- all six of CoReg's user-facing presets from `utils/presets.py`
   are available (`random` is excluded: it's only an internal empty-result
   fallback, not a real preset):
   - `regular_grid` / `regular_grid_spacing` -- uniform coverage
   - `shi_tomasi` -- corner/feature detection (OpenCV `goodFeaturesToTrack`)
   - `motion` / `greedy_motion` -- seeded from inter-frame intensity change
   - `segmentation` -- centroids of thresholded connected components
   - `matchanything` -- learned keypoint matching (falls back to
     `shi_tomasi` if the optional `transformers` model isn't available)
3. Click "Run CoReg". Two new layers are added: the motion-corrected video
   and a `Tracks` layer with the tracked points.
