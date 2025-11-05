These are independent project files for evaluating traditional Non-NN models on downstream tasks. We used supercomputer (HPC) to run the project.

Folder descriptions:
-- logs -> where to save ur project notice from HPC
-- models -> the definition of every Non-NN models
-- results -> where to save ur results

Practical guidance:
Step1:
You need to config the following .sh files to get the results on downstream tasks:
- run_ML_decoders_for_restTasks.sh
- run_ML_decoders_for_SSVEP.sh

step2:
Submit those .sh files to HPC. The you get the saved results in the 'results' folder.