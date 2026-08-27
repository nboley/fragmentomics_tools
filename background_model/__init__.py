"""Background model v2 — data plumbing.

Preprocess cfDNA fragment h5 files into a zarr store suitable for training
the sequence-driven background model (see background_model_core.py).
"""

from background_model.config import PlumbingConfig
