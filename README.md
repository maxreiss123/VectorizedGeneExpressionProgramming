# VectorizedGeneExpressionProgramming for symbolic regression
Dealing with high-dimensional data often causes evolutionary methods to struggle. Due to modern computational power, a simple first step increases hyperparameters, such as the population size or the number of epochs. 

At a certain point, the algorithm faces a high workload caused by the internal representation of the objects. Object-orientated data structures allow a highly readable and logical structure. Still, when it comes to a considerable number of replication procedures by utilizing genetic operators, the overhead of computational time and memory allocations becomes non-negligible. 

Within that version, we tokenize the symbols and achieve a genotype representation where arithmetic operations can be applied easily. The given repo implements two versions, one in Julia and one in Python, whereby the Gene-Expression-Programming[1] serves as a role model. 


- Velocity check: 200.000 datapoints, 1000 candidate solutions, mating size 0.4, four threads
  - Julia: 69s
  - Python: 635s
 
- Amount of data points tested without issues - time scales linear
  - 1e5
  - 1e6
  - 1e7

- Amount of candidate solutions (population size) without issues
  - 1e4
  - 1e5
  - 1e6 -> Remark: initialization takes a while
  - 1e7 -> Remark: initialization takes a while

# Todo
- Try to enforce the constraint to max work on float32
- Inner approximation of the constants with perturbation and the application of the gene fusion operator
- Random application of genetic operators
- Conversion to Julia - (Needs to be tested/debugged)
- Implementing of learnable weights for the genetic operators
- Implementation of an attention-based mutation operator
- Automatic gene-len scaling
- Further operators like RIS, IS
- Further selection mechanisms
- NSGA2

#Remarks
- Please add some comments

# How to use it?
- In your terminal, open the JuliaREPL and enter ```]```
- After the package manager pops up, add the following packages:
  ```
  add DynamicExpressions Statistics LinearAlgebra ProgressMeter 
  ``` 
- Clone the repository:
  ```git clone https://github.com/maxreiss123/VectorizedGeneExpressionProgramming.git```
- Than you can run:
  ```julia main_min_example.jl```

