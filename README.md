## MCANet: A joint semantic segmentation framework of optical and SAR images for land use classification
[Paper address](https://www.sciencedirect.com/science/article/pii/S0303243421003457)
### [WHU-SAR-OPT DataSet](https://github.com/AmberHen/WHU-OPT-SAR-dataset)

##### Make DataSet WHU-OPT-SAR 
```Python3
# Cut SAR Image into 256x256 patches
python utils/crop_all_sar.py
# Cut Optical Image into 256x256 patches
python utils/crop_all_opt.py
# Convert Label RGB to [0:7]
python utils/convert_label.py
# Cut Optical Image into 256x256 patches
python utils/crop_all_lbl.py
# Spilt data into train/validation/test set    6:2:2
python utils/spilt_data.py
```
######  Dataset Structure
```
dataset 
|   |whu-opt-sar-dataset    Totals
│   ├──   ├── train         17640
│   ├──   │     ├── sar     
│   ├──   │     │     ├── *.tif   
│   ├──   │     ├── opt
│   ├──   │     │     ├── *.tif   
│   ├──   │     ├── lbl
│   ├──   │     │     ├── *.tif 
│   ├──   ├── val           5880
│   ├──   │     ├── sar     
│   ├──   │     │     ├── *.tif
│   ├──   │     ├── opt
│   ├──   │     │     ├── *.tif
│   ├──   │     ├── lbl
│   ├──   │     │     ├── *.tif
│   ├──   ├── test           5880
│   ├──   │     ├── sar
│   ├──   │     │     ├── *.tif
│   ├──   │     ├── opt
│   ├──   │     │     ├── *.tif
│   ├──   │     ├── lbl
│   ├──   │     │     ├── *.tif
```
