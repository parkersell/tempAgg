## Forked from Original TempAgg code

Code: https://github.com/parkersell/tempAgg/tree/main

F. Sener, D. Singhania and A. Yao, "Temporal Aggregate Representations for Long-Range Video Understanding", ECCV 2020 [paper]

F. Sener, D. Chatterjee and A. Yao, "Technical Report: Temporal Aggregate Representations", arXiv:2106.03152, 2021 [paper]

## Setup data

Setup data to be in this folder configuration. YOUR_DATA_PATH can be just tempAgg/data/

```
$YOUR_DATA_PATH
├── annots/ # found from https://github.com/assembly-101/assembly101-mistake-detection
|   ├──nusar-2021_action_both_9011-b06b_9011_user_id_2021-02-01_154253.csv
|   ├──...
├── TSM_features/
|   ├──C10119_rgb/  # you can add other views if you want to test those, we found this one to work fine
|   |   ├── data.mdb
|   │   ├── lock.mdb
├── infoTempAgg.json
```


## Setup environment

`conda create env -f environment.yml`

And change COMP_PATH [here](https://github.com/parkersell/tempAgg/blob/mistake_detection/main_mistake_detection.py#L24).

## Running
`python main_mistake_detection.py`