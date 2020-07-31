# TVI-FL

This repository contains code that implements the Time-Varying Ising model with Fused and Lasso penalties (TVI-FL) algorithm presented in the paper:

Learning the piece-wise constant graph structure of a varying Ising model. Batiste Le Bars, Pierre Humbert, Argyris Kalogeratos and Nicolas Vayatis. In Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

The paper is available at https://proceedings.icml.cc/static/paper_files/icml/2020/2583-Paper.pdf

I. Contents

- ICML_Experiment.ipynb
- TVI_FL.py
- ChangePoints.npy
- LearnedGraphs.npy
- Party_info.csv
- Train.csv

II. Code Information

- Use the TVI_FL function from TVI_FL.py to apply the proposed algorithm from the paper "Learning the piece-wise constant graph structure of a varying Ising model".

- To perform the real-world experiment on the votes in the llinois House of Representatives, run the Jupyter Notebook "ICML_Experiment.ipynb". The data are already preprocessed.
  
- If code cell 4 is too long to run, then you can directly run cell 5 to obtain the results.
