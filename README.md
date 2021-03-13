#CDS-FTR:  Carton datasets synthesis based on foreground texture replacement
texture replacement.

# install
python 3.5

OpenCV (pip install opencv-python)

PIL (pip install Pillow)

Poisson Blending (Follow instructions https://github.com/yskmt/pb)

PyBlur (pip install pyblur)

# Running the Script
Test faceted results：

python test_get_surface.py -bf ./data/back -sf ./data/surface

Test construction results:

python test_construction_surface.py -bf ./data/back -sf ./data/construction

the final results:

pyhon main.py -ff ./data/fore/img -bf ./data/back -sf ./data/save





