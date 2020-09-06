# Introduction of projects


## Comments_Classification

Aim:

     This project will implement several core parts of the stacking machine learning method in Pyspark. 
     More specifically, you are required to complete a series of functions with methods in PySpark SQL and PySpark MLlib to identify the category of each customer review using the review text and the trained model.


Description:


     1. The dataset consists of sentences from customer reviews of different restaurants. There are 2241, 800, 800 customer reviews in the train, development, and test datasets, respectively. It should be noted that there is at least one sentence in each customer review and each customer review may not be with ending punctuation such as `.`, `?`.

     2. The categories include:
         * FOOD: reviews that involve comments on the food. 
             - e.g. “All the appetizers and salads were fabulous , the steak was mouth watering and the pasta was delicious”
         * PAS: reviews that only involve comments on price, ambience, or service. 
             - e.g. “Now it 's so crowded and loud you ca n't even talk to the person next to you”
         * MISC: reviews that do not belong to the above categories including sentences that are general recommendations  reviews describing the reviewer’s personal experience or context, but that do not usually provide information on the restaurant quality 
             - e.g. “Your friends will thank you for introducing them to this gem!”
             - e.g. “I knew upon visiting NYC that I wanted to try an original deli”
     3. You can view samples from the dataset using `dataset.show()` to get five samples with `descript` column showing the review text and `category` column showing the annotated class.


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

