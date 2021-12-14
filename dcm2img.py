# Reference: https://github.com/ucs198604/dicom2jpg
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut

def dcm2img(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array.astype(float)

    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        rescale_slope = float(ds.RescaleSlope) 
        rescale_intercept = float(ds.RescaleIntercept)
        pixel_array = (pixel_array) * rescale_slope + rescale_intercept
    else:
        pixel_array = apply_modality_lut(pixel_array, ds)

    if 'VOILUTFunction' in ds and ds.VOILUTFunction=='SIGMOID':
        pixel_array = apply_voi_lut(pixel_array, ds)
    elif 'WindowCenter' in ds and 'WindowWidth' in ds:
        window_center = ds.WindowCenter
        window_width = ds.WindowWidth
        if type(window_center)==pydicom.multival.MultiValue:
            window_center = float(window_center[0])
        else:
            window_center = float(window_center)
        if type(window_width)==pydicom.multival.MultiValue:
            window_width = float(window_width[0])
        else:
            window_width = float(window_width)
        pixel_array = np.piecewise(
            pixel_array, [pixel_array<=(window_center-(window_width)/2), pixel_array>(window_center+(window_width)/2)], 
            [pixel_array.min(), pixel_array.max(), lambda data: ((data-window_center+window_width/2)/window_width*(pixel_array.max() - pixel_array.min()))+pixel_array.min()])
    else:
        pixel_array = apply_voi_lut(pixel_array, ds)

    # normalize to 8 bit
    pixel_array = ((pixel_array-pixel_array.min())/(pixel_array.max()-pixel_array.min())) * 255.0

    if 'PhotometricInterpretation' in ds and ds.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = np.max(pixel_array) - pixel_array

    image = pixel_array.astype('uint8')

    return image