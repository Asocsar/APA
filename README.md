# APA Project Development
### By Daniel Cano Carrascosa and Eduard Bosch i Mustarós


## Getting the data
Although all the needed data is in the data folder in the root of this repository, you can download it following this tutorial:
The data is obtained from the UCI repository through this link: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
We have to go to the Data folder (in the download section of said link) and download the following files:
  - cleveland.data
  - hungarian.data
  - long-beach-va.data
  - new.data
  - switzerland.data

## Preprocessing

Before we start using the models, we have to execute our preprocessing script, that will create a new csv called preprocess.csv and store in the data folder. 
To execute the preprocessing we will have to launch a python notebook and navigate to the script folder, there we will find "Preprocessing.ipynb", that we will have to execute in order to pre process all the data.

## Executing a model

At this point, we are able to execute any model in our script folder. To do so, you just have to execute a notebook and navigate to whichever model you want to execute. Bear in mind that the SVM (the linear one mainly) takes a long time to execute with the C parameters that we have in our scripts, so we would recommend uncommenting the line that discards the biggest C and use that instead of the ones we used.

