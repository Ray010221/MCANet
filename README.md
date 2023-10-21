## MCANet: A joint semantic segmentation framework of optical and SAR images for land use classification
[Paper address](https://www.sciencedirect.com/science/article/pii/S0303243421003457)
***
I am not the author of the paper, and my code is a replication of the model based on the information provided in the paper. It is an unofficial implementation intended for learning and reference purposes. If you have any questions, feel free to contact me via email(1256154030@qq.com).
***
### [WHU-SAR-OPT DataSet](https://github.com/AmberHen/WHU-OPT-SAR-dataset)
##### Make DataSet WHU-OPT-SAR 
```Linux
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

##### Bash
```Linux
nohup python3 -u Train.py > train-MCANet.log 2>&1 &
pyhton3 predict.py
python3 test.py     # get OA on test dataset
```