import numpy
from PIL import Image, ImageDraw


def polygon_to_mask(img_size, polygon):
    # Takes polygon points as list of tuples (x, y) and creates np.array of size(x, y)
    # if pixel is inside polygon value of pixel is set to 1

    img = Image.new("L", img_size, 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    return numpy.array(img)
