"""DE-CLIP: Description-Enhanced CLIP for zero-shot dermatology image classification.

The package is organised along the two stages of the method:

* ``encode`` -- frozen CLIP encoders turn images and prompts into embeddings (cached).
* ``aggregate`` -- the training-free refinement of Sec. 3.5-3.6 that turns the
  per-class prompt embeddings into one class text embedding.
"""

__version__ = "1.0.0"
