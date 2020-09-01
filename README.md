# Introduction of projects

## (OpenCV) cell detection, tracking, and analysis

Aim:

     Automate the detection and tracking of the cells in time-lapse microscopy images to support different biological experiments or observations.


Description:

     The image data to be used in the group project is taken from the international Cell Tracking Challenge (CTC)[1].
     The developed methods will work on all these data.

   

## Fuzzy Scheduling


Aim:

     This project concerns developing optimal solutions to a scheduling problem inspired by the scenario of a manufacturing plant 
     that has to fulfil multiple customer orders with varying deadlines, but there may be constraints on tasks and relationships between tasks.

Description:

     This assignment is an example of a constraint optimization problem.
     That has constraints like a standard Constraint Satisfaction Problem (CSP), but also a cost as-sociated with each solution.
     I implemented a greedy algorithm to find optimal solutions to fuzzy scheduling problems(if a solution exists).

CSP:

     1. a set of variables representing tasks, such as task, 〈name〉 〈duration〉
        eg: task, t1 3 
            task, t2 4 

     2. binary constraints on pairs of tasks
        eg:  constraint, 〈t1〉 before 〈t2〉       t1 ends when or before t2 starts 

     3. unary constraints (hard or soft) on tasks.
        eg:  domain, 〈t〉 〈day〉 # t starts on given day at any time 
   
   
## Sentiment Analysis

Aims: 

    Analyse the Twitter feed to determine customer sentiment towards *** company and its competitors.

Description:

     In this assignment, there will be given a collection of tweets about US airlines. 
     The tweets have been manually labelled for sentiment. Sentiment is categorized as either positive, negative or neutral. 
     Note: These tweets as dataset will not be uploaded on the Internet, as this breaches Twitter’s Terms of Service.




[1] V. Ulman et al. An objective comparison of cell-tracking algorithms. Nature Methods, vol. 14, no. 2, pp.1141-1152, December 2017. https://doi.org/10.1038/nmeth.4473

