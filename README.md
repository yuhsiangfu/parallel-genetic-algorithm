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
> -graph: att48_d <br>
> 
> -serial <br>
> --final evo-num:  1 <br>
> --final evo-fit:  49926 <br>
> --final evo-path: [44, 34, 25, 3, 9, 23, 41, 1, 28, 4, 47, 38, 31, 20, 24, 22, 40, 21, 2, 0, 15, 33, 13, 12, 46, 32, 7, 8, 10, 11, 35, 16, 27, 30, 39, 14, 18, 29, 19, 45, 43, 17, 37, 6, 5, 26, 42, 36] <br>
> -396.6317 sec. <br>
> 
> -parallel <br>
> --final evo-num:  1 <br>
> --final evo-fit:  43838 <br>
> --final evo-path: [44, 34, 23, 25, 3, 9, 41, 1, 28, 4, 47, 38, 31, 46, 20, 19, 11, 39, 12, 15, 40, 33, 24, 22, 13, 10, 2, 21, 7, 30, 37, 14, 0, 8, 45, 32, 35, 6, 43, 17, 29, 27, 26, 5, 36, 18, 16, 42] <br>
> -234.0715 sec. <br>
 
## Notification
1. You are free to use the codes for educational purposes.
2. Our coding style may not as good as you expect, but it works.
3. We are glad to hear your improvements of the codes.
4. Any questions please contact yuhisnag.fu@gmail.com.

Best regards,
Yu-Hsiang Fu 2022418 updated.
