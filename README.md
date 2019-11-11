## ReinforceML
A handy Data Science Assistant for beginners and exerts alike. ReinforceML is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming and reinforcement learning.


  
# ReinforceML: An informed Evolutionary Approach to Automated Machine Learning 

Multi-Objective Evolutionary Algorithms(MOEA) are population-based stochastic or meta-heuristic optimization algorithms. MOEAs involve several competing objectives and a set of Pareto optimal solutions. MOEAs are able to estimate Pareto optimal set in a single run by evolving a population of solutions.

We model the machine learning pipeline as an individual in a population. It is a tree-based pipeline optimization tool with machine learning transformers as nodes of a tree and their respective parameters as terminals.  At the moment, we have implemented an informed evolutionary approach to AutoML. Our hybrid approach is a combination of MOEA and Reinforcement learning (RL). It employs MOEA to generate a population of self-sufficient machine learning pipelines to solve the problem of AFE while RL assists MOEA to make informed decisions during evolution. Gradient information is crucial for informed evolution since it exerts a selection pressure towards those regions with an expected higher return. 


All the transformers such as Scalers, Selectors, Classifiers, etc are derived from sklearn. we build a lookup of sklearn estimators along with the possible hyper-parameters. We envision to extend it to Neural Architecture Search (NAS) eventually.  
