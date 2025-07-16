# PiX: PDE System Interpretable eXplorer

This is a user-friendly platform for interpretable PDE system discovery directly from observational data and physics prior knowledge, serving as an easy-to-use platform based on the idea of the PhysPDE [paper] (https://openreview.net/forum?id=G3CpBCQwNh) and an extension to the official [code] (https://github.com/FengMingquan-sjtu/PhysPDE). 

## Overview
PhysPDE proposes a new pipeline for PDE interepretation based on hypotheses selection and symbolic regression. Here's the pipeline from the original paper.

However, the original implementation requires deep knowledge of the codebase to add new physics assumptions, limiting practical use. This new platform is rebuilt, which
**1. Enables Direct and Easy Use for Physicists:** Physicists can model a complicated physics field with this tool by providing observational data and physics prior knowledge by just preparing a simple csv file and don't need to modify the source code.
**2. Accelerates Research for AI Researchers:** AI researchers who are interested in this novel task can directly apply their methods into the pipeline and tested without rewriting the code. This is particularly helpful for researchers that are not familiar with sympy, which the platform relies heavily on.

## Platform
Testing Platform: Ubuntu22.04 & macOS Sequoia 15.5.
Current code is not available for Windows OS due to signal.SIGALRM module.

## Manual for Physicists

### 1. Build Your csv File or Use Existing yaml
The csv file should be formatted as: (an example is also given in `physics_problem.csv`)
1. Problem Name & Brief Description  
2. Variable Names  
3. Constant Inputs  
4. Derived Physical Quantities & Definition Equations  
   • (Mark as "=?" if definition is unknown or requires hypothesis enumeration. Ensure all variables up to this point are declared via variables, constants, or definitions.)  

5. Predefined Physical Laws  
6. Decision Tree Input (Per Line Format):  
   • (Hypothesis ID, Hypothesis Name, Parent Node, Requires Symbolic Regression?,  

     ◦ Expression (eq: <expression>),  

     ◦ Definition (def: [new_var_1; new_var_2; ...] var=...),  

     ◦ Constraint (constraint: [var]...>0 / [var]...>=0),  

     ◦ Variable Relationships (related: var1; var2; var3; var4))  

7. Symbolic Regression Function Library  
   • (Specify whether nested functions are allowed.)  

After building it and naming it 'physics_problem.csv', run `physics_config_helper.py` and get the config. Please put your dataset in `pix/data`.

### 2. Quick Test
```bash
conda create -n pix python=3.9
conda activate pix
pip install -r requirements.txt
```
Then modify the `pix/cfg/config.yaml` to choose your problem and the search algorithm. Finally, run `main.py`.

## Manual for AI Researchers

### 1. Quick Learn
`calculator.py`: The core ​symbolic computation framework​ which automatically manages physical quantities and equation. Once the object is constructed, it'll load the data and the expression of every physical quantity and equation. You can update the dictionaries by just using the `update_unknown_var` function, and realize automatic elimination by calling `update_local_dict` function.

`hypotheses_tree.py`: The tree structure storing all the hypotheses. Every single tree has a Calculator in it. Later it'll be extended to graphs as decision trees are no longer sufficient for modeling more complicated systems.

`data_loader.py`: This is automatically called by `calculator.py`. Now only supporting grid data.

`utils`: Numerical computation tools for sympy and numpy are stored here.

`methods`: You can put your algorithms here.

### 2. Datasets
The datasets that the original authors put out can be accessed from [here](https://zenodo.org/records/11530771?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjgwZGFkMTJiLTY2NjYtNGY0MS04YzI4LTZkMzRjNmM2ZGRlZCIsImRhdGEiOnt9LCJyYW5kb20iOiI0NTA4MWU5MzkxOGU4YjYwNjdlMGJkYmUzY2NmYjM5YSJ9.50vo70qCuAfIokz6KsUps-DaQbppGM75joD8DpyLi-6lVn3DGgtTDzv6MSgRx2wl9RmTi8T1yjx785gHJuEyvA). Please put the data in `pix/data` 

### 3. Build your csv file for More Test
Please follow the **Manual for Physicists** to include more tests.
