import nbformat, sys, pathlib
nb_path = pathlib.Path("YouTube-ASL Clip Keypoint Dataset/YouTubeASL keypoints demo.ipynb")   # adjust
nb      = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
nb.metadata.pop("widgets", None)        # remove stale widget metadata
nbformat.write(nb, nb_path)
print("🧹  metadata.widgets removed")