Download ``EuroSAT_RGB.zip`` from the official repo: [https://github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT) and extract the contents here.

Expected directory structure:

```
.
├── AnnualCrop
├── Forest
├── HerbaceousVegetation
├── Highway
├── Industrial
├── Pasture
├── PermanentCrop
├── Residential
├── River
├── SeaLake
├── create_txt_files_eurosat.ipynb
└── README.md
```

Run the [create_txt_files_eurosat.ipynb](create_txt_files_eurosat.ipynb) notebook (re)generate the following label files in [../../benchmark_imglist/eurosat](../../benchmark_imglist/eurosat): 
* eurosat_test_ood.txt