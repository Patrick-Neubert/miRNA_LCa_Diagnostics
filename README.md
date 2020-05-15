# miRNA Biomarker for Lung Cancer Diagnostics
***- Selecting a test panel for patient classification -***

---

## Overview

This repository contains the code, visualizations and presentations of my capstone project which was part of my professional training as Data Scientist at neuefische GmbH (https://www.neuefische.de/).

## Abstract

Lung cancer is the most prevalent cancer disease worldwide. The current standard diagnostics for lung cancer are X-ray imaging, sputum cytology and tissue samples (biopsy), which are inapplicable for early detection screenings. My project uses machine learning algorithms analyzing a panel of miRNA biomarkers in blood samples to classify patients into the groups "lung cancer", "non-tumor lung disease" and "control".

## Data

Hummingbird Diagnostics GmbH (https://www.hummingbird-diagnostics.com) uses liquid biopsy diagnostics (such as blood samples) to detect diseases based on highly disease-specific miRNA biomarkers. The data was kindly provided via neuefische GmbH and originally used for developping a biochip for lung cancer diagnostics, whereby only 10-20 miRNAs need to be tested.

## Project goal and objectives

The goal of this EDA project is to identify the 10-20 most important biomarkers for classification of patients, which achieve the best results for a chosen metric score.

## Technologie Highlights

Python / Pandas / scikit-learn / NumPy / Matplotlib / Seaborn / Feature Selection / Ensemble Methods

## Files and Folders

### data 
- Contains the Lung Cancer miRNA Expression Data and the additional Annotation Data from Hummingbird Diagnostics GmbH 
- Contains a pickle file with intermediate results from model optimization with RandomizedSearch and GridSearchCV
- Contains X_tbc_top20, X_tbc_top20_test, y_test, y_train dataframes from main notebook to be used in the outsourced 
  RandomizedSearch and GridSearchCV notebook

### figures
- Contains all figures used in the notebook or graphs ploted in the notebook
 
### presentations
- Contains a .pdf file with final presentation
 
### functions.ipynb
- all functions written for the notebook

### functions.py
- all functions written for the notebook (accessible as .py-file)

### conda_enviroment.yml
- conda enviroment with all packages and modules used in the notebook

### mirna_biomarker_for_lung_cancer_diagnostics.ipynb
- notebook with Exploratory Data Analysis (EDA), python code, visualizations, literature research, documentation and project report

### random_and_grid_search.ipynb
- notebook with RandomizedSearch and GridSearchCV outsourced from main notebook to save time when re-computing. 











