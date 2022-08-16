#CDS-FTR:  
@article{gou2022carton,
  title={Carton dataset synthesis method for loading-and-unloading carton detection based on deep learning},
  author={Gou, Lijun and Wu, Shengkai and Yang, Jinrong and Yu, Hangcheng and Lin, Chenxi and Li, Xiaoping and Deng, Chao},
  journal={The International Journal of Advanced Manufacturing Technology},
  pages={1--18},
  year={2022},
  publisher={Springer}
}

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





