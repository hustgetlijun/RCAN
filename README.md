#CDS-FTR:  Carton dataset synthesis based on foreground texture replacement.

# install
python 3.5

OpenCV (pip install opencv-python)

PIL (pip install Pillow)

Poisson Blending (Follow instructions https://github.com/yskmt/pb)

PyBlur (pip install pyblur)

# Running the Script
Test surface segmentation：

python test_get_surface.py -bf ./data/back -sf ./data/surface

Test contour constructed:

python test_construction_surface.py -bf ./data/back -sf ./data/construction

the total results:

pyhon main.py -ff ./data/fore/img -bf ./data/back -sf ./data/save





