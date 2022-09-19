# wadda_mlops
> Wind's MLOps for autonomous driving perception tasks

## Introduction
> Introduction about Autonomous Driving Perception Tasks

The autonomous driving perception task should be a closed-loop task, and its life cycle should include at least three stages, such as data, model, and deployment. Therefore, in the design process of MLOps, at least these three directions need to be covered. More features will be added in the future

## Data
The importance of data for deep learning is self-evident, and a good dataset should take into account both "depth" and "breadth". The breadth can be satisfied with the help of open source data sets and its own continuous collection, while the depth needs to be combined with actual tests, targeted analysis, and re-collection to complete.

### Dataset
- Open source dataset
     - Collection: Collect and download data sets related to autonomous driving
     - Analysis: Do data analysis on the collected data set, and evaluate whether the data distribution it contains meets the needs of the current task
     - Summarize and organize: For useful open source data, classify it according to the category of tasks, and manage it together with the data set of the same task
     - Conversion: According to the needs of the pipeline, develop the corresponding conversion script and convert it to the data set format required by the pipeline
- custom data
     - Data set definition: Customize the required data according to the needs of the task, including its category, labeling standard, labeling format, etc.
     - Data collection: develop data collection tools, which need to meet the above definition requirements, and need to consider the needs of data screening, continuous collection and other tasks
     - Data annotation: According to the definition, write detailed annotation requirements, and then hand it over to the data annotation company
     - Analysis, induction, transformation: refer to the description in the open source dataset

### Dataset Iteration
It is impossible to solve all scene problems by collecting data several times, because the corner cases in the automatic driving task are difficult to solve at once, so it is necessary to continuously collect new data according to the actual test situation to supplement the existing ones. data set to gradually cover as many scenes as possible

### Dataset Pipeline Design
With the development of the autonomous driving industry, there will inevitably be more models with better performance, and it is very likely that the way these models load data will be different from the previous general methods, such as the popular BEV scheme not long ago. Therefore, in order to adapt to these models, it is necessary to write the corresponding pipeline for loading data, so that the training and testing of the new model can be quickly completed based on the existing data set.

## Modeling
> The model part is based on a series of excellent frameworks open-sourced by the open-mmlab laboratory. It can easily reproduce some classic models in the past, and has built a new model based on it to verify its own. idea, thanks to open-mmlab

### Model Reproduction & Framework Extension & Network Refactoring
According to information such as papers and open source codes, some classic and cutting-edge models in the field of autonomous driving are reproduced in MLOps. For some relatively new or not mainstream network structures, open-mmlab has not yet supported them. These networks The structure needs to be implemented manually, and I usually write these implementations in the extension module. In addition, if you want to verify some of your ideas, you can refactor the model structure by simply modifying the configuration file, so as to quickly complete some experiments

### Train & Test & Evaluate
It is convenient to train the model and help us evaluate the effect of the model and the effect of the dataset through the results of testing and evaluation

## Deploy
The deployment of the model is a complete engineering problem, and the biggest problem in deployment is often caused by the deployment operator provided by the deployment platform used does not support some operators in the deployment network.

Even so, we still try our best to base on some mature deployment frameworks, such as tensorrt, openvino, etc., which will greatly simplify the deployment task.

If you want to further optimize the deployment, you also need deployment skills such as knowledge distillation and pruning.