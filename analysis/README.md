To reproduce the figures & tables from the paper, run the [paper_figures.ipynb](paper_figures.ipynb) notebook. You can also visualize the notebook using NBViewer [here](https://nbviewer.org/github/glhr/ood-labelnoise/blob/release/analysis/paper_figures.ipynb). 

The raw results used as a basis for the paper are in the [result_csvs]([result_csvs]) folder. These CSV files were generated from `update_allresults_file.py`, which loads the .pkl files of individual models/OOD detectors from [../results/](../results/) and aggregates them into a Pandas dataframe. 
