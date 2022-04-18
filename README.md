# Parallel-Genetic-Algorithm
This is an exercise of applying `multiprocessing` of Python to the genetic algorithm and solving the travelling salesman problem (TSP).

## Setting of the execution environment
In this excercise, I used Python 3.7.10, NetworkX 2.6.3 and Numpy 1.21.2 for programming. Also, for the IDE of Python, we recommend the PyCharm community version.

## Dataset
The dataset used in this exercise, please see the url: https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html.

## Simple code usage
For TSP datasets:
1. set parameters of the code `parallel-genetic-algorithm-v1-2.py`, then run it. The parameters are as follows.
  `graph_name = "att48_d"` <br>
  `num_cpu = 8` <br>
  `num_evolution = 1` <br>
  `num_iteration = 2000` <br>
  `size_population = 2000` <br>
  `rate_selection = 0.2` <br>
  `rate_crossover = 0.8` <br>
  `rate_mutation = 0.05` <br>

The results of the above parameters of "serial-ga" and "parallel-ga", please see the below information.
>
 
## Notification
1. You are free to use the codes for educational purposes.
2. Our coding style may not as good as you expect, but it works.
3. We are glad to hear your improvements of the codes.
4. Any questions please contact yuhisnag.fu@gmail.com.

Best regards,
Yu-Hsiang Fu 2022418 updated.
