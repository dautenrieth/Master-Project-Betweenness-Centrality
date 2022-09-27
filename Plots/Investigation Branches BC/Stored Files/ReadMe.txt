Guidance for reproducing content
-------------------------------------------------------------------------------------------------------------------
This folder contains all files to reproduce the graphs from the Investigation Branches BC folder. Since the error distribution changes slightly with each run.
The sampled points from which the bounding functions are obtained must be picked again after running the approximation algortihms.
To use exactly the same run, the files must be copied into the respective folders:
1. The Approx files go into the Approx_Betweenness folder.
2. The error files go into the Errors folder and 
3. the saved points must be copied back into the GetNodeBranches.py file (if the points saved in it have been changed).
Then simply run the GetNodeBranches.py file again.