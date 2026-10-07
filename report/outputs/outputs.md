
# Counting vehicles by class for signal timing (UVH-26, YOLOv8)


### cell 1: `# ---- Run settings: the only cell a grader should need to edit ----`


### cell 2: `import os, subprocess, sys`

```
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
$ git fetch -q origin
$ git checkout -q -f --detach origin/main
$ grep -v colab-skip requirements.txt > /tmp/req-colab.txt && pip install -q -r /tmp/req-colab.txt
repo: /content/uvh26-pcu-counting | commit: 6d8234e
```

### cell 3: `import torch`

```
seed: 42
  python           3.13.15
  numpy            2.1.3
  matplotlib       3.10.0
  torch            2.11.0+cu130
  torchvision      0.26.0+cu130
  opencv-python    5.0.0.93
  transformers     5.18.0
  ultralytics      8.4.173
  huggingface-hub  1.33.0
  pandas           2.2.3
  pyyaml           6.0.3
  tqdm             4.67.3
device: Tesla T4
paths: {'raw_root': '/content/raw', 'data_root': '/content/drive/MyDrive/uvh26/prepared', 'labels_root': '/content/uvh26-pcu-counting/data/labels', 'runs_root': '/content/drive/MyDrive/uvh26/runs', 'weights_dir': '/content/drive/MyDrive/uvh26/weights', 'datasets_root': '/content/datasets', 'splits_dir': '/content/uvh26-pcu-counting/splits', 'results_dir': '/content/drive/MyDrive/uvh26/results', 'tags_file': '/content/uvh26-pcu-counting/analysis/phase6_tags.csv'}
```

## 1. Problem and task type


## 2. Data and split  *(Phase 1)*


### cell 6: `import pandas as pd`

```
pool: 4000 images with both MV and STAPLE labels
official_split  subfolder
train           000          785
                001          711
                002          768
                003          732
                004          205
val             000          743
                001           56
pool 4000 images, 4000 already prepared, 0 to download
```

### cell 8: `prep = prepare.run(cfg, pool)`

```

```
```
{'images': 4000, 'official_split': {'train': 3201, 'val': 799}, 'output_size_wh': [1280, 720], 'jpeg_total_mb': 1431.4, 'boxes': {'mv': 48932, 'st': 51780}, 'boxes_per_image': {'mv': 12.23, 'st': 12.95}, 'exact_duplicate_boxes_dropped': {'mv': 0, 'st': 429}}
```
```
                    MV     ST
Hatchback         4664   4765
Sedan             2443   2894
SUV               1874   2450
MUV                938   1599
Bus               1422   1410
Truck             2140   2836
Three-wheeler     7895   7783
Two-wheeler      23628  23400
LCV               2675   2918
Mini-bus           130    181
Tempo-traveller    205    228
Bicycle            574    901
Van                319    376
Others              25     39
```
![cell08_3.jpg](cell08_3.jpg)

### Pseudo-cameras and the official split's leakage


### cell 10: `clu = cluster.run(cfg)`

```
Using frozen /content/uvh26-pcu-counting/splits/groups.csv; recomputed grouping identical: True
linking rule: g >= 0.48 | 3083 pseudo-cameras | 2703 singletons | largest 65 images
calibration (equal cameras): 2129 groups, 24.0% singletons
LEAKAGE: 247 of 799 official-val images (30.9%) share a pseudo-camera with an official-train image
frozen groups used; recomputed grouping identical: True
```
```
                                          rule  groups  \
setting                                                  
calibration       Poisson, 2,800 equal cameras    2129   
A_tau0.90                         CLIP >= 0.90     856   
A_tau0.92                         CLIP >= 0.92    2245   
A_tau0.94                         CLIP >= 0.94    3595   
A_tau0.96                         CLIP >= 0.96    3901   
B_tau0.88_r0.6      CLIP >= 0.88 AND r >= 0.60    3016   
B_tau0.88_r0.7      CLIP >= 0.88 AND r >= 0.70    3385   
B_tau0.88_r0.8      CLIP >= 0.88 AND r >= 0.80    3689   
B_tau0.90_r0.6      CLIP >= 0.90 AND r >= 0.60    3198   
B_tau0.90_r0.7      CLIP >= 0.90 AND r >= 0.70    3461   
B_tau0.90_r0.8      CLIP >= 0.90 AND r >= 0.80    3718   
B_tau0.92_r0.6      CLIP >= 0.92 AND r >= 0.60    3442   
B_tau0.92_r0.7      CLIP >= 0.92 AND r >= 0.70    3586   
B_tau0.92_r0.8      CLIP >= 0.92 AND r >= 0.80    3766   
C_r0.40                              r >= 0.40     553   
C_r0.40_clip0.80    CLIP >= 0.80 AND r >= 0.40     723   
C_r0.45                              r >= 0.45     942   
C_r0.45_clip0.80    CLIP >= 0.80 AND r >= 0.45    1117   
C_r0.50                              r >= 0.50    1418   
C_r0.50_clip0.80    CLIP >= 0.80 AND r >= 0.50    1618   
C_r0.55                              r >= 0.55    1906   
C_r0.55_clip0.80    CLIP >= 0.80 AND r >= 0.55    2122   
C_r0.60                              r >= 0.60    2426   
C_r0.60_clip0.80    CLIP >= 0.80 AND r >= 0.60    2570   
G_g0.30                              g >= 0.30    1110   
G_g0.35                              g >= 0.35    1800   
G_g0.40                              g >= 0.40    2417   
G_g0.45                              g >= 0.45    2894   
G_g0.50                              g >= 0.50    3186   
G_g0.55                              g >= 0.55    3355   
G_g0.60                              g >= 0.60    3484   

                  singletons_pct_of_pool  leakage_pct     edges  \
setting                                                           
calibration                         24.0         68.1       NaN   
A_tau0.90                           19.9         79.2   33438.0   
A_tau0.92                           51.1         47.3    3968.0   
A_tau0.94                           84.4         14.5     545.0   
A_tau0.96                           95.7          4.0     111.0   
B_tau0.88_r0.6                      66.0         33.0    2127.0   
B_tau0.88_r0.7                      77.0         22.1    1088.0   
B_tau0.88_r0.8                      87.5         11.6     421.0   
B_tau0.90_r0.6                      71.2         27.4    1571.0   
B_tau0.90_r0.7                      79.4         19.7     885.0   
B_tau0.90_r0.8                      88.5         10.4     368.0   
B_tau0.92_r0.6                      79.0         20.2     955.0   
B_tau0.92_r0.7                      84.1         15.4     609.0   
B_tau0.92_r0.8                      90.3          8.9     292.0   
C_r0.40                             12.7         89.6  131604.0   
C_r0.40_clip0.80                    16.4         85.9   67297.0   
C_r0.45                             21.3         80.1   62828.0   
C_r0.45_clip0.80                    24.8         77.2   33739.0   
C_r0.50                             31.5         68.6   28478.0   
C_r0.50_clip0.80                    35.5         63.9   16418.0   
C_r0.55                             41.7         58.8   12660.0   
C_r0.55_clip0.80                    45.9         54.4    8008.0   
C_r0.60                             53.4         46.1    5641.0   
C_r0.60_clip0.80                    56.1         43.5    4089.0   
G_g0.30                             23.8         76.0   63217.0   
G_g0.35                             38.2         62.1   17894.0   
G_g0.40                             51.0         48.3    6057.0   
G_g0.45                             62.2         36.4    3173.0   
G_g0.50                             70.7         27.7    2149.0   
G_g0.55                             76.2         23.4    1507.0   
G_g0.60                             80.8         18.3     988.0   

                  largest_group  largest_group_pct_of_pool  d_groups  \
setting                                                                
calibration                 NaN                        NaN       NaN   
A_tau0.90                3010.0                       75.2   -1273.0   
A_tau0.92                1415.0                       35.4     116.0   
A_tau0.94                  33.0                        0.8    1466.0   
A_tau0.96                   8.0                        0.2    1772.0   
B_tau0.88_r0.6            170.0                        4.2     887.0   
B_tau0.88_r0.7             39.0                        1.0    1256.0   
B_tau0.88_r0.8             21.0                        0.5    1560.0   
B_tau0.90_r0.6             58.0                        1.5    1069.0   
B_tau0.90_r0.7             32.0                        0.8    1332.0   
B_tau0.90_r0.8             20.0                        0.5    1589.0   
B_tau0.92_r0.6             41.0                        1.0    1313.0   
B_tau0.92_r0.7             30.0                        0.8    1457.0   
B_tau0.92_r0.8             19.0                        0.5    1637.0   
C_r0.40                  3388.0                       84.7   -1576.0   
C_r0.40_clip0.80         3189.0                       79.7   -1406.0   
C_r0.45                  2931.0                       73.3   -1187.0   
C_r0.45_clip0.80         2704.0                       67.6   -1012.0   
C_r0.50                  2347.0                       58.7    -711.0   
C_r0.50_clip0.80         2066.0                       51.6    -511.0   
C_r0.55                  1707.0                       42.7    -223.0   
C_r0.55_clip0.80         1416.0                       35.4      -7.0   
C_r0.60                  1093.0                       27.3     297.0   
C_r0.60_clip0.80          873.0                       21.8     441.0   
G_g0.30                  2631.0                       65.8   -1019.0   
G_g0.35                  1632.0                       40.8    -329.0   
G_g0.40                   823.0                       20.6     288.0   
G_g0.45                   292.0                        7.3     765.0   
G_g0.50                    57.0                        1.4    1057.0   
G_g0.55                    50.0                        1.2    1226.0   
G_g0.60                    50.0                        1.2    1355.0   

                  d_singletons_pp  calib_distance  
setting                                            
calibration                   NaN             NaN  
A_tau0.90                    -4.1           0.622  
A_tau0.92                    27.1           1.130  
A_tau0.94                    60.4           2.609  
A_tau0.96                    71.7           3.101  
B_tau0.88_r0.6               42.0           1.799  
B_tau0.88_r0.7               53.0           2.286  
B_tau0.88_r0.8               63.5           2.745  
B_tau0.90_r0.6               47.2           2.030  
B_tau0.90_r0.7               55.4           2.392  
B_tau0.90_r0.8               64.5           2.789  
B_tau0.92_r0.6               55.0           2.373  
B_tau0.92_r0.7               60.1           2.596  
B_tau0.92_r0.8               66.3           2.868  
C_r0.40                     -11.3           0.877  
C_r0.40_clip0.80             -7.6           0.732  
C_r0.45                      -2.7           0.569  
C_r0.45_clip0.80              0.8           0.477  
C_r0.50                       7.5           0.457  
C_r0.50_clip0.80             11.5           0.536  
C_r0.55                      17.7           0.745  
C_r0.55_clip0.80             21.9           0.913  
C_r0.60                      29.4           1.233  
C_r0.60_clip0.80             32.1           1.353  
G_g0.30                      -0.2           0.479  
G_g0.35                      14.2           0.612  
G_g0.40                      27.0           1.133  
G_g0.45                      38.2           1.632  
G_g0.50                      46.7           2.008  
G_g0.55                      52.2           2.250  
G_g0.60                      56.8           2.451  
```

### cell 11: `rule = cluster.Rule.from_config(cfg).name`

![cell11_0.png](cell11_0.png)
![cell11_1.png](cell11_1.png)
![cell11_2.jpg](cell11_2.jpg)
![cell11_3.jpg](cell11_3.jpg)
![cell11_4.jpg](cell11_4.jpg)
![cell11_5.jpg](cell11_5.jpg)

### Group split and class balance


### cell 13: `if (cfg["paths"]["splits_dir"] / "test.sha256").exists():`

```
split is frozen; test.txt sha256 verified: f8181d247d7e034f5810b2bb2a978e2739e5055dbc8b3240877c5caa1f667ff6
images: {'train': 3071, 'val': 452, 'test': 477}
image_share: {'train': 0.7678, 'val': 0.113, 'test': 0.1192}
pseudo_cameras: {'train': 2318, 'val': 355, 'test': 410}
official_split_composition: {'train': {'train': 2477, 'val': 594}, 'val': {'train': 356, 'val': 96}, 'test': {'train': 368, 'val': 109}}
classes_missing: {'mv/val': [], 'mv/test': [], 'st/val': [], 'st/test': []}
classes_under_30_test_instances: {'mv': {'Mini-bus': 21, 'Tempo-traveller': 26, 'Others': 2}, 'st': {'Mini-bus': 25, 'Tempo-traveller': 28, 'Others': 3}}
```
```
                    mv                 st            
                 train   val  test  train   val  test
Hatchback         3463   615   586   3539   619   607
Sedan             1829   311   303   2159   379   356
SUV               1371   262   241   1803   347   300
MUV                666   141   131   1164   223   212
Bus               1053   221   148   1051   212   147
Truck             1624   269   247   2147   350   339
Three-wheeler     5916  1019   960   5840  1006   937
Two-wheeler      17780  3002  2846  17603  2981  2816
LCV               2047   285   343   2217   329   372
Mini-bus            91    18    21    138    18    25
Tempo-traveller    162    17    26    179    21    28
Bicycle            475    50    49    709    97    95
Van                230    41    48    270    44    62
Others              21     2     2     32     4     3
```

## 3. Preprocessing and augmentation  *(Phases 1-2)*


## 4. Baseline: YOLOv8s fine-tuned on MV labels  *(Phase 2)*


### cell 16: `from src import evaluate, train`

```
MV labels: mAP50 0.274, mAP50-95 0.181 over all coarse classes | mappable only: mAP50 0.493, mAP50-95 0.326
ST labels: mAP50 0.243, mAP50-95 0.163 over all coarse classes | mappable only: mAP50 0.438, mAP50-95 0.294
```
```
                   AP50  AP50_95  instances
car              0.8385   0.6679     1329.0
bus              0.5858   0.4594      239.0
truck            0.4803   0.3506      554.0
motorcycle       0.3768   0.1197     3002.0
bicycle          0.1827   0.0299       50.0
three-wheeler    0.0000   0.0000     1019.0
tempo-traveller  0.0000   0.0000       17.0
van              0.0000   0.0000       41.0
others           0.0000   0.0000        2.0
```

### Freeze pilot (val only)


### cell 18: `if RUN_TRAINING:`

```
pilot_frozen_mv: finished
pilot_full_mv: finished
```
```
                       frozen     full
val mAP50-95           0.4054   0.4325
val mAP50              0.4946   0.5194
minutes / epoch        1.7200   1.8700
first epoch (min)      1.7800   1.8800
total (min)            8.6800   9.3800
epochs within 90 min  37.0000  34.0000
```

### Epoch budget


### cell 20: `t = cfg["train"]`

```
config: train.freeze = None, train.epochs = 34 (close_mosaic 10, mosaic cost x1.35, budget 90 min)
  frozen: freeze = 10, 1.72 min/epoch without mosaic -> 37 epochs fit the budget (worst case); 34 epochs take at most 73.0 min
  full  : freeze = None, 1.87 min/epoch without mosaic -> 34 epochs fit the budget (worst case); 34 epochs take at most 79.3 min
```

### Baseline run


### cell 22: `if RUN_TRAINING:`

```
mv: finished
{'epochs_run': 34, 'minutes_total': 65.44, 'minutes_per_epoch': 1.92}
val during training, best epoch: {'epoch': 34, 'precision': 0.6913, 'recall': 0.5524, 'mAP50': 0.6142, 'mAP50_95': 0.524}
```
![cell22_1.png](cell22_1.png)

### Generalisation gap


### cell 24: `best = baseline_dir / "weights" / "best.pt"`

```
mv__train__mv: unchanged weights and settings, using the saved evaluation
mv__val__mv: unchanged weights and settings, using the saved evaluation
```
```
           train    val    gap
mAP50      0.914  0.614  0.300
mAP50_95   0.824  0.524  0.300
precision  0.873  0.692  0.181
recall     0.826  0.554  0.272
```
```
                 train    val  val instances    gap
AP50-95                                            
Hatchback        0.866  0.570            615  0.296
Sedan            0.874  0.557            311  0.318
SUV              0.862  0.477            262  0.385
MUV              0.842  0.411            141  0.431
Bus              0.889  0.702            221  0.187
Truck            0.863  0.569            269  0.294
Three-wheeler    0.885  0.782           1019  0.103
Two-wheeler      0.829  0.710           3002  0.119
LCV              0.864  0.578            285  0.286
Mini-bus         0.771  0.502             18  0.270
Tempo-traveller  0.870  0.621             17  0.248
Bicycle          0.818  0.525             50  0.294
Van              0.835  0.326             41  0.509
Others           0.474  0.014              2  0.459
```

### Fine-tuning gain, like for like


### cell 26: `fine = evaluate.finetuned_coarse(cfg, best, "mv")`

```
                                COCO as-is  fine-tuned on MV
mAP50-95, all 9 coarse classes       0.181             0.552
mAP50-95, 5 COCO-mappable            0.326             0.663
mAP50, all 9 coarse classes          0.274             0.659
mAP50, 5 COCO-mappable               0.493             0.805
```
```
                 COCO as-is  fine-tuned on MV
AP50-95                                      
car                   0.668             0.802
bus                   0.459             0.663
truck                 0.351             0.632
motorcycle            0.120             0.707
bicycle               0.030             0.511
three-wheeler         0.000             0.773
tempo-traveller       0.000             0.638
van                   0.000             0.247
others                0.000             0.000
```

## 5. Ablation: MV vs STAPLE training labels  *(Phase 3)*


### cell 28: `if RUN_TRAINING:`

```
st: finished
{'epochs_run': 34, 'minutes_total': 66.31, 'minutes_per_epoch': 1.94}
val during training (ST labels), best epoch: {'epoch': 25, 'precision': 0.6429, 'recall': 0.5772, 'mAP50': 0.5844, 'mAP50_95': 0.494}
```
![cell28_1.png](cell28_1.png)

### Proof of control


### cell 30: `ctrl = train.args_diff(cfg, "mv", "st")`

```
                               MV run                       STAPLE run
data  /content/datasets/uvh26_mv.yaml  /content/datasets/uvh26_st.yaml
name                               mv                               st
```
```
label-set keys: ['data'] | bookkeeping: ['name'] | unexpected: []
same train / val images: {'train': True, 'val': True} | resumed: {'mv': False, 'st': False}
ONLY DIFFERENCE IS THE LABEL SET: True
```

### 2 × 2: trained-on × scored-against (val)


### cell 32: `runs = {"mv": baseline_dir, "st": ablation_dir}`

```
mv__val__mv: unchanged weights and settings, using the saved evaluation
mv__val__st: unchanged weights and settings, using the saved evaluation
st__val__mv: unchanged weights and settings, using the saved evaluation
st__val__st: unchanged weights and settings, using the saved evaluation
mAP50_95
```
```
               scored vs MV labels  scored vs ST labels
trained on MV               0.5245               0.4923
trained on ST               0.5131               0.4948
```
```
mAP50
```
```
               scored vs MV labels  scored vs ST labels
trained on MV               0.6145               0.5779
trained on ST               0.6107               0.5850
```
```
precision
```
```
               scored vs MV labels  scored vs ST labels
trained on MV               0.6919               0.6280
trained on ST               0.5203               0.6428
```
```
recall
```
```
               scored vs MV labels  scored vs ST labels
trained on MV               0.5538               0.5778
trained on ST               0.6506               0.5761
```
```
                 MV-trained vs MV  MV-trained vs ST  ST-trained vs MV  \
AP50-95                                                                 
Hatchback                   0.570             0.572             0.543   
Sedan                       0.557             0.582             0.525   
SUV                         0.477             0.512             0.514   
MUV                         0.411             0.394             0.345   
Bus                         0.702             0.702             0.697   
Truck                       0.569             0.504             0.550   
Three-wheeler               0.782             0.788             0.779   
Two-wheeler                 0.710             0.706             0.696   
LCV                         0.578             0.539             0.566   
Mini-bus                    0.502             0.375             0.429   
Tempo-traveller             0.621             0.582             0.622   
Bicycle                     0.525             0.295             0.559   
Van                         0.326             0.330             0.310   
Others                      0.014             0.011             0.050   

                 ST-trained vs ST  val instances (MV)  
AP50-95                                                
Hatchback                   0.533                 615  
Sedan                       0.564                 311  
SUV                         0.534                 262  
MUV                         0.361                 141  
Bus                         0.700                 221  
Truck                       0.504                 269  
Three-wheeler               0.783                1019  
Two-wheeler                 0.693                3002  
LCV                         0.531                 285  
Mini-bus                    0.378                  18  
Tempo-traveller             0.632                  17  
Bicycle                     0.323                  50  
Van                         0.348                  41  
Others                      0.044                   2  
```

### Is the difference larger than evaluation noise?


### cell 34: `boot = {sc: evaluate.paired_bootstrap(cfg, f"mv__val__{sc}", f"st__val__{sc}", n_boot=cfg[`

```
                                     scored vs MV labels scored vs ST labels
MV-trained mAP50-95                               0.5245              0.4923
ST-trained mAP50-95                               0.5131              0.4948
ST - MV, mAP50-95                                -0.0114              0.0025
95% CI                                 [-0.0304, 0.0135]    [-0.0131, 0.024]
ST - MV, mAP50                                   -0.0038              0.0071
95% CI                                 [-0.0274, 0.0347]     [-0.0101, 0.03]
resamples where ST better (mAP50-95)               0.218                0.62
```

### Which classes move?


### cell 36: `per_cls = {}`

```
                ST - MV vs ST        95% CI vs ST ST - MV vs MV  \
AP50-95                                                           
Hatchback             -0.0393  [-0.0646, -0.0093]       -0.0276   
Sedan                 -0.0181   [-0.0504, 0.0149]       -0.0315   
SUV                    0.0212   [-0.0106, 0.0576]        0.0366   
MUV                    -0.033   [-0.0759, 0.0224]       -0.0657   
Bus                   -0.0027   [-0.0272, 0.0207]       -0.0048   
Truck                 -0.0001    [-0.0315, 0.032]       -0.0191   
Three-wheeler         -0.0052    [-0.013, 0.0033]       -0.0031   
Two-wheeler           -0.0124  [-0.0174, -0.0077]       -0.0133   
LCV                   -0.0084   [-0.0389, 0.0246]       -0.0122   
Mini-bus               0.0029   [-0.1525, 0.1501]       -0.0729   
Tempo-traveller        0.0504    [-0.048, 0.1564]         0.001   
Bicycle                0.0282    [-0.0118, 0.071]        0.0339   
Van                    0.0185   [-0.0927, 0.1166]       -0.0158   
Others                 0.0329   [-0.0058, 0.2808]        0.0351   

                       95% CI vs MV  val instances (ST)  
AP50-95                                                  
Hatchback         [-0.0569, 0.0031]                 619  
Sedan              [-0.0615, 0.004]                 379  
SUV               [-0.0063, 0.0808]                 347  
MUV               [-0.1207, 0.0196]                 223  
Bus                [-0.026, 0.0198]                 212  
Truck              [-0.054, 0.0195]                 350  
Three-wheeler     [-0.0107, 0.0059]                1006  
Two-wheeler      [-0.0184, -0.0086]                2981  
LCV               [-0.0424, 0.0231]                 329  
Mini-bus          [-0.2518, 0.0708]                  18  
Tempo-traveller    [-0.102, 0.1127]                  21  
Bicycle           [-0.0361, 0.1134]                  97  
Van               [-0.1327, 0.0924]                  44  
Others            [-0.0097, 0.2961]                   4  
```

### Robustness: last-epoch weights


### cell 38: `last = {(tr, sc): evaluate.detector_metrics(cfg, runs[tr] / "weights" / "last.pt", sc, "va`

```
mv_last__val__mv: unchanged weights and settings, using the saved evaluation
mv_last__val__st: unchanged weights and settings, using the saved evaluation
st_last__val__mv: unchanged weights and settings, using the saved evaluation
st_last__val__st: unchanged weights and settings, using the saved evaluation
```
```
                     scored vs MV labels scored vs ST labels
MV-trained, epoch 34              0.5245              0.4923
ST-trained, epoch 34              0.5101              0.4869
ST - MV, mAP50-95                -0.0144             -0.0053
95% CI                 [-0.0319, 0.0112]   [-0.0213, 0.0186]
```

## 6. Evaluation  *(Phase 4 threshold on val; Phase 5 sealed test, once)*


### From detections to Passenger Car Units  *(Phase 4, val only)*


### cell 41: `from src import counts`

```
                IRC category  PCU factor  category share of train traffic (%)
Hatchback                car      1.0000                                23.00
Sedan                    car      1.0000                                23.00
SUV                      car      1.0000                                23.00
MUV                      car      1.0000                                23.00
Van                      car      1.0000                                23.00
Bus                truck_bus      3.2760                                 8.59
Truck              truck_bus      3.2760                                 8.59
Mini-bus           truck_bus      3.2760                                 8.59
Tempo-traveller          lcv      1.5401                                 6.17
LCV                      lcv      1.5401                                 6.17
Three-wheeler           auto      2.0000                                15.03
Two-wheeler      two_wheeler      0.7500                                45.31
Bicycle                cycle      0.4000                                 1.82
Others               tractor      4.0000                                 0.08
```

### Operating threshold


### cell 43: `cnt = counts.run(cfg)`

```
val: 452 images, mean true demand 18.482 PCU per image (ST labels)
```
![cell43_1.png](cell43_1.png)
```
                   trained on MV  trained on ST
chosen threshold          0.3500         0.3500
frozen in config          0.3500         0.3500
pcu_mae vs ST             2.7358         2.7081
pcu_bias vs ST           -0.5025        -0.8830
pcu_rel_mae vs ST         0.1480         0.1465
count_mae vs ST           1.9004         1.9690
count_bias vs ST         -0.0288        -0.3230
pcu_mae vs MV             2.3510         2.2559
pcu_bias vs MV            0.6853         0.3047
pcu_rel_mae vs MV         0.1359         0.1304
count_mae vs MV           1.7876         1.7279
count_bias vs MV          0.8053         0.5111
```
```
MV-trained model at threshold 0.35: per-class counts vs ST labels
```
```
                 true per image  count MAE  signed bias  true total  \
Hatchback                 1.369      0.715       -0.011       619.0   
Sedan                     0.838      0.407       -0.093       379.0   
SUV                       0.768      0.507       -0.175       347.0   
MUV                       0.493      0.367       -0.239       223.0   
Bus                       0.469      0.179        0.024       212.0   
Truck                     0.774      0.471       -0.228       350.0   
Three-wheeler             2.226      0.456        0.150      1006.0   
Two-wheeler               6.595      1.310        0.686      2981.0   
LCV                       0.728      0.425        0.013       329.0   
Mini-bus                  0.040      0.031       -0.022        18.0   
Tempo-traveller           0.046      0.046        0.011        21.0   
Bicycle                   0.215      0.168       -0.124        97.0   
Van                       0.097      0.093       -0.013        44.0   
Others                    0.009      0.009       -0.009         4.0   

                 predicted total  
Hatchback                  614.0  
Sedan                      337.0  
SUV                        268.0  
MUV                        115.0  
Bus                        223.0  
Truck                      247.0  
Three-wheeler             1074.0  
Two-wheeler               3291.0  
LCV                        335.0  
Mini-bus                     8.0  
Tempo-traveller             26.0  
Bicycle                     41.0  
Van                         38.0  
Others                       0.0  
```
```
ST-trained model at threshold 0.35: per-class counts vs ST labels
```
```
                 true per image  count MAE  signed bias  true total  \
Hatchback                 1.369      0.728       -0.215       619.0   
Sedan                     0.838      0.409       -0.122       379.0   
SUV                       0.768      0.447       -0.009       347.0   
MUV                       0.493      0.376       -0.252       223.0   
Bus                       0.469      0.190        0.004       212.0   
Truck                     0.774      0.427       -0.192       350.0   
Three-wheeler             2.226      0.414        0.060      1006.0   
Two-wheeler               6.595      1.314        0.597      2981.0   
LCV                       0.728      0.394       -0.035       329.0   
Mini-bus                  0.040      0.029       -0.029        18.0   
Tempo-traveller           0.046      0.031        0.004        21.0   
Bicycle                   0.215      0.173       -0.111        97.0   
Van                       0.097      0.113       -0.015        44.0   
Others                    0.009      0.009       -0.009         4.0   

                 predicted total  
Hatchback                  522.0  
Sedan                      324.0  
SUV                        343.0  
MUV                        109.0  
Bus                        214.0  
Truck                      263.0  
Three-wheeler             1033.0  
Two-wheeler               3251.0  
LCV                        313.0  
Mini-bus                     5.0  
Tempo-traveller             23.0  
Bicycle                     47.0  
Van                         37.0  
Others                       0.0  
```

### Sealed test set  *(Phase 5, evaluated once)*


### cell 45: `if not EVALUATE_TEST:`

```
test.txt sha256 verified: f8181d247d7e034f5810b2bb2a978e2739e5055dbc8b3240877c5caa1f667ff6 | frozen thresholds: {'mv': 0.35, 'st': 0.35}
mv__test__mv: unchanged weights and settings, using the saved evaluation
mv__test__st: unchanged weights and settings, using the saved evaluation
st__test__mv: unchanged weights and settings, using the saved evaluation
st__test__st: unchanged weights and settings, using the saved evaluation
TEST mAP50_95
```
```
               scored vs MV labels  scored vs ST labels
trained on MV               0.4892               0.4550
trained on ST               0.4750               0.4482
```
```
TEST mAP50
```
```
               scored vs MV labels  scored vs ST labels
trained on MV               0.5710               0.5324
trained on ST               0.5545               0.5277
```
```
TEST precision
```
```
               scored vs MV labels  scored vs ST labels
trained on MV               0.6156               0.6090
trained on ST               0.5997               0.6167
```
```
TEST recall
```
```
               scored vs MV labels  scored vs ST labels
trained on MV               0.5619               0.5137
trained on ST               0.5510               0.5045
```
```
                    TEST, scored vs MV labels TEST, scored vs ST labels
MV-trained mAP50-95                    0.4892                     0.455
ST-trained mAP50-95                     0.475                    0.4482
ST - MV, mAP50-95                     -0.0141                   -0.0068
95% CI                       [-0.035, 0.0105]         [-0.0226, 0.0078]
ST - MV, mAP50                        -0.0165                   -0.0047
95% CI                      [-0.0378, 0.0114]         [-0.0246, 0.0125]
```
```
                 MV-trained vs MV  MV-trained vs ST  ST-trained vs MV  \
TEST AP50-95                                                            
Hatchback                   0.579             0.579             0.556   
Sedan                       0.587             0.596             0.578   
SUV                         0.439             0.450             0.430   
MUV                         0.385             0.352             0.352   
Bus                         0.654             0.661             0.650   
Truck                       0.554             0.462             0.515   
Three-wheeler               0.798             0.800             0.794   
Two-wheeler                 0.734             0.727             0.714   
LCV                         0.587             0.552             0.571   
Mini-bus                    0.257             0.181             0.159   
Tempo-traveller             0.484             0.444             0.448   
Bicycle                     0.427             0.248             0.430   
Van                         0.353             0.298             0.385   
Others                      0.009             0.020             0.068   

                 ST-trained vs ST  test instances (ST)  
TEST AP50-95                                            
Hatchback                   0.566                  607  
Sedan                       0.586                  356  
SUV                         0.483                  300  
MUV                         0.349                  212  
Bus                         0.641                  147  
Truck                       0.464                  339  
Three-wheeler               0.796                  937  
Two-wheeler                 0.710                 2816  
LCV                         0.551                  372  
Mini-bus                    0.128                   25  
Tempo-traveller             0.432                   28  
Bicycle                     0.266                   95  
Van                         0.299                   62  
Others                      0.003                    3  
```

#### Generalisation: train, val and test


### cell 47: `if EVALUATE_TEST:`

```
mv__train__mv: unchanged weights and settings, using the saved evaluation
st__train__st: unchanged weights and settings, using the saved evaluation
```
```
                          train    val   test  train - test  val - test
mAP50-95                                                               
trained and scored on MV  0.824  0.524  0.489         0.335       0.035
trained and scored on ST  0.660  0.495  0.448         0.212       0.047
```

#### Counts and PCU at the frozen thresholds


### cell 49: `if EVALUATE_TEST:`

```
test: mean true demand per image (PCU): {'mv': 15.351, 'st': 16.484}
```
```
                   trained on MV @ 0.35  trained on ST @ 0.35
pcu_mae vs ST                    2.6092                2.7002
pcu_bias vs ST                  -0.9349               -1.2521
pcu_rel_mae vs ST                0.1583                0.1638
count_mae vs ST                  1.7526                1.8050
count_bias vs ST                -0.4990               -0.7107
pcu_mae vs MV                    2.2943                2.2904
pcu_bias vs MV                   0.1990               -0.1182
pcu_rel_mae vs MV                0.1495                0.1492
count_mae vs MV                  1.5849                1.5702
count_bias vs MV                 0.2306                0.0189
```
```
TEST, MV-trained at 0.35: per-class counts vs ST labels
```
```
                 true per image  count MAE  signed bias  true total  \
Hatchback                 1.273      0.623        0.048       607.0   
Sedan                     0.746      0.417       -0.115       356.0   
SUV                       0.629      0.405       -0.166       300.0   
MUV                       0.444      0.321       -0.187       212.0   
Bus                       0.308      0.149       -0.031       147.0   
Truck                     0.711      0.440       -0.168       339.0   
Three-wheeler             1.964      0.390        0.067       937.0   
Two-wheeler               5.904      0.981        0.222      2816.0   
LCV                       0.780      0.478        0.071       372.0   
Mini-bus                  0.052      0.052       -0.036        25.0   
Tempo-traveller           0.059      0.048       -0.010        28.0   
Bicycle                   0.199      0.180       -0.122        95.0   
Van                       0.130      0.122       -0.067        62.0   
Others                    0.006      0.006       -0.006         3.0   

                 predicted total  
Hatchback                  630.0  
Sedan                      301.0  
SUV                        221.0  
MUV                        123.0  
Bus                        132.0  
Truck                      259.0  
Three-wheeler              969.0  
Two-wheeler               2922.0  
LCV                        406.0  
Mini-bus                     8.0  
Tempo-traveller             23.0  
Bicycle                     37.0  
Van                         30.0  
Others                       0.0  
```
```
TEST, ST-trained at 0.35: per-class counts vs ST labels
```
```
                 true per image  count MAE  signed bias  true total  \
Hatchback                 1.273      0.667       -0.159       607.0   
Sedan                     0.746      0.423       -0.155       356.0   
SUV                       0.629      0.430        0.010       300.0   
MUV                       0.444      0.342       -0.191       212.0   
Bus                       0.308      0.136       -0.027       147.0   
Truck                     0.711      0.398       -0.159       339.0   
Three-wheeler             1.964      0.369        0.021       937.0   
Two-wheeler               5.904      0.979        0.216      2816.0   
LCV                       0.780      0.392       -0.031       372.0   
Mini-bus                  0.052      0.050       -0.042        25.0   
Tempo-traveller           0.059      0.042       -0.017        28.0   
Bicycle                   0.199      0.182       -0.094        95.0   
Van                       0.130      0.122       -0.075        62.0   
Others                    0.006      0.006       -0.006         3.0   

                 predicted total  
Hatchback                  531.0  
Sedan                      282.0  
SUV                        305.0  
MUV                        121.0  
Bus                        134.0  
Truck                      263.0  
Three-wheeler              947.0  
Two-wheeler               2919.0  
LCV                        357.0  
Mini-bus                     5.0  
Tempo-traveller             20.0  
Bicycle                     50.0  
Van                         26.0  
Others                       0.0  
```

#### Report figures: precision-recall and confusion


### cell 51: `if EVALUATE_TEST:`

![cell51_0.png](cell51_0.png)
![cell51_1.png](cell51_1.png)
![cell51_2.png](cell51_2.png)
![cell51_3.png](cell51_3.png)
![cell51_4.png](cell51_4.png)

## 7. Error analysis  *(Phase 6)*


### cell 53: `if EVALUATE_TEST:`

```
MV-trained @ 0.35 vs ST labels: the 30 worst of 477 test images carry 24% of the total absolute PCU error
```
```
        stem           picked_by  pcu_true  pcu_pred  pcu_err  abs_pcu_err  \
rank                                                                         
1     785597  PCU error + missed    27.058    10.750  -16.308       16.308   
2       1904  missed + PCU error    54.342    39.500  -14.842       14.842   
3     217500  missed + PCU error    55.592    45.052  -10.540       10.540   
4     416055  PCU error + missed    25.158    12.276  -12.882       12.882   
5     853707  missed + PCU error    33.196    23.330   -9.866        9.866   
6     845401  PCU error + missed    30.948    18.330  -12.618       12.618   
7     148550           PCU error    23.868    36.448   12.580       12.580   
8     772122              missed    67.187    71.071    3.884        3.884   
9     137148  PCU error + missed    37.066    25.066  -12.000       12.000   
10    163766              missed    43.658    46.488    2.830        2.830   
11    186004  PCU error + missed    36.132    24.330  -11.802       11.802   
12    984665           PCU error    37.828    26.500  -11.328       11.328   
13    102998  missed + PCU error    32.144    21.092  -11.052       11.052   
14    527484  PCU error + missed    68.210    57.052  -11.158       11.158   
15    856265  missed + PCU error    47.816    38.540   -9.276        9.276   
16     98960           PCU error    16.526    27.368   10.842       10.842   
17    549996  PCU error + missed    40.448    29.646  -10.802       10.802   
18    434201              missed    36.282    28.632   -7.650        7.650   
19    838491              missed    22.106    15.080   -7.026        7.026   
20    497937  PCU error + missed    19.052     8.750  -10.302       10.302   
21    989345              missed    37.014    30.278   -6.736        6.736   
22    689478              missed    28.466    24.316   -4.150        4.150   
23    798180           PCU error    20.632    10.790   -9.842        9.842   
24    558051           PCU error    11.396    21.170    9.774        9.774   
25    861563           PCU error    15.302     5.750   -9.552        9.552   
26      5342           PCU error    33.514    43.052    9.538        9.538   
27    362092           PCU error    36.802    27.500   -9.302        9.302   
28     22788              missed    29.444    21.436   -8.008        8.008   
29    664051              missed    18.276    11.000   -7.276        7.276   
30    971007           PCU error    21.144    12.052   -9.092        9.092   

      vehicles_true  vehicles_pred  missed  wrong_class  false_pos  \
rank                                                                 
1                19             10      10            2          1   
2                49             38      14            7          3   
3                46             36      11            4          1   
4                16             10       7            2          1   
5                28             20      11            3          3   
6                22             17       7            2          2   
7                15             23       1            1          9   
8                46             47      10            3         11   
9                28             19       9            2          0   
10               31             31      10            7         10   
11               31             23       8            1          0   
12               29             24       5            9          0   
13               21             15       9            2          3   
14               46             43       8           10          5   
15               41             33       9            5          1   
16               15             20       1            0          6   
17               27             21       7            1          1   
18               30             25       8            3          3   
19               22             18       8            0          4   
20               15              9       7            0          1   
21               26             21       8            2          3   
22               23             18       8            1          3   
23               14              9       5            2          0   
24                8             10       1            4          3   
25               11              6       5            1          0   
26               22             26       2            1          6   
27               27             23       4            1          0   
28               21             17       7            7          3   
29               20             14       7            2          1   
30               11              7       5            0          1   

      pcu_from_wrong_class  missed_small  missed_at_edge  boxes_mv  night_ir  
rank                                                                          
1                    1.000             0               1        15     False  
2                   -5.552             0               1        44     False  
3                    0.000             0               0        43     False  
4                   -0.540             0               2        12     False  
5                    1.600             0               2        21     False  
6                   -1.736             0               2        19     False  
7                    0.000             0               0        14     False  
8                   -0.196             0               1        43     False  
9                    0.000             0               1        26     False  
10                   1.250             0               0        30     False  
11                   0.000             0               3        30     False  
12                  -6.828             0               2        28     False  
13                   0.000             0               1        19     False  
14                  -5.632             0               1        43     False  
15                  -2.276             0               3        38     False  
16                   0.000             0               0        14     False  
17                  -1.736             0               1        27     False  
18                   0.540             0               4        29     False  
19                   0.000             0               2        19     False  
20                   0.000             0               0        11     False  
21                  -1.986             0               2        22     False  
22                   0.000             0               2        21     False  
23                   0.000             0               1        13     False  
24                   4.958             0               1         8     False  
25                  -2.276             0               2        10     False  
26                   1.736             0               0        21     False  
27                   0.000             0               1        26     False  
28                  -2.932             0               1        18     False  
29                   0.000             0               1        17     False  
30                   0.000             0               1         9     False  
```

### cell 54: `if EVALUATE_TEST:`

![cell54_0.jpg](cell54_0.jpg)
![cell54_1.jpg](cell54_1.jpg)
![cell54_2.jpg](cell54_2.jpg)
![cell54_3.jpg](cell54_3.jpg)
![cell54_4.jpg](cell54_4.jpg)
![cell54_5.jpg](cell54_5.jpg)
![cell54_6.jpg](cell54_6.jpg)
![cell54_7.jpg](cell54_7.jpg)
![cell54_8.jpg](cell54_8.jpg)
![cell54_9.jpg](cell54_9.jpg)
![cell54_10.jpg](cell54_10.jpg)
![cell54_11.jpg](cell54_11.jpg)
![cell54_12.jpg](cell54_12.jpg)
![cell54_13.jpg](cell54_13.jpg)
![cell54_14.jpg](cell54_14.jpg)
![cell54_15.jpg](cell54_15.jpg)
![cell54_16.jpg](cell54_16.jpg)
![cell54_17.jpg](cell54_17.jpg)
![cell54_18.jpg](cell54_18.jpg)
![cell54_19.jpg](cell54_19.jpg)
![cell54_20.jpg](cell54_20.jpg)
![cell54_21.jpg](cell54_21.jpg)
![cell54_22.jpg](cell54_22.jpg)
![cell54_23.jpg](cell54_23.jpg)
![cell54_24.jpg](cell54_24.jpg)
![cell54_25.jpg](cell54_25.jpg)
![cell54_26.jpg](cell54_26.jpg)
![cell54_27.jpg](cell54_27.jpg)
![cell54_28.jpg](cell54_28.jpg)
![cell54_29.jpg](cell54_29.jpg)

### Tagging and tally


### cell 56: `if EVALUATE_TEST:`

```
No tags yet: phase6_tags.csv
```

## 8. Conclusion
