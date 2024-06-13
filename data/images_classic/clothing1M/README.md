Download the Clothing1M dataset by following the instructions in the official repo: https://github.com/Cysu/noisy_label

Expected directory structure:
```shell
.
├── annotations
│   ├── category_names_chn.txt
│   ├── category_names_eng.txt
│   ├── clean_label_kv.txt
│   ├── clean_test_key_list.txt
│   ├── clean_train_key_list.txt
│   ├── clean_val_key_list.txt
│   ├── noisy_label_kv.txt
│   └── noisy_train_key_list.txt
├── images
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   ├── 5
│   ├── 6
│   ├── 7
│   ├── 8
│   ├── 9
│   └── clean_subset.zip
├── calc_normalization_params.py
├── create_txt_files_clothing1m.ipynb
└── README.md
```

Run the [create_txt_files_clothing1m.ipynb](create_txt_files_clothing1m.ipynb) notebook to (re)generate the following label files in [../../benchmark_imglist/clothing1M](../../benchmark_imglist/clothing1M):
* train_clothing1M_clean.txt
* train_clothing1M_noisy.txt
* val_clothing1M_clean.txt
* test_clothing1M_clean.txt
