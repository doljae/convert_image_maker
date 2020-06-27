from image.image_handle import convertImageMaker
import os
import glob
import random

files = glob.glob('./image/crop_images/*')
for f in files:
    os.remove(f)

convertTest=convertImageMaker()
convertTest.image_extract()
convertTest.image_save_crop_location()
convertTest.image_convert(convert_type=3)
# convertTest.image_convert(convert_type=random.randint(1,4))
