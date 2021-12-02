import PIL.ExifTags


def formatted_exif_data(image):
    """Retrieve an image's EXIF data and return as a dictionary with string keys"""
    # with thanks to https://stackoverflow.com/questions/4764932/in-python-how-do-i-read-the-exif-data-for-an-image
    exif_data = {}
    for k, v in image._getexif().items():
        # Use names rather than codes where they are available
        try:
            exif_data[PIL.ExifTags.TAGS[k]] = v
        except KeyError:
            exif_data[k] = v
    return exif_data


def getExposureTime(exif_data):
    if "ExposureTime" in exif_data:
        return float(exif_data["ExposureTime"])
    elif "ShutterSpeedValue" in exif_data:
        ssv = exif_data["ShutterSpeedValue"]
        print(ssv)
        return 1 / 2.0 ** (float(ssv[0]) / ssv[1])
    else:
        raise Exception("No exposure time found!")
