from fileinput import filename
import pathlib
from typing import List
# from visual import Annotator
import napari
from PIL import Image
from skimage import data


def get_filename_paths() -> List[pathlib.Path]:
    path = pathlib.Path("references/object-detection/cards/input/")
    filenames = [file for file in path.glob("*") if file.is_file()]
    return filenames


if __name__ == "__main__":
    filenames = get_filename_paths()
    # image = Image.open(filenames[0])
    viewer = napari.view_image(data.astronaut(), rgb=True)
    napari.run()  # start the event loop and show viewerd