# Environment Setup 

Before running everything, create a new conda environment with python 3.10 installed 
```
conda env create -n "myEnv" python=3.10
conda activate myEnv 
pip install torch torchvision matplotlib seaborn umap-learn umap-learn[plot] xgboost scikit-learn nltk
conda install numpy pandas 
```

Please refer to `preprocess.py` for feature extraction. 

We implement 5 models on the Fall 23 CS671 Airbnb Dataset
1) KNN Classifier 
2) MLP Classifier 
3) MLP Regression with Output Clipping 
4) Random Forest
5) XGBoost 

