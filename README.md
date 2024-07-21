# VectorizedGeneExpressionProgramming for symbolic regression
Dealing with high-dimensional data often causes evolutionary methods to struggle. Due to modern computational power, a simple first step increases hyperparameters, such as the population size or the number of epochs. 

At a certain point, the algorithm faces a high workload caused by the internal representation of the objects. Object-orientated data structures allow a highly readable and logical structure. Still, the overhead of computational time and memory allocations becomes non-negligible when it comes to many replication procedures that utilize genetic operators. 

Within that version, we tokenize the symbols and achieve a genotype representation where arithmetic operations can be applied easily. The given repo implements a version in Julia, whereby the Gene-Expression-Programming [1] serves as a role model. 
 
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
- [ ] Try to enforce the constraint to max work on float32
- [x] Inner approximation of the constants with perturbation and the application of the gene fusion operator
- [x] Random application of genetic operators
- [ ] Conversion to Julia - (Needs to be tested/debugged)
- [ ] Implementing learnable weights for the genetic operators
- [ ] Implementation of an attention-based mutation operator
- [ ] Automatic gene-len scaling
- [x] Further operators like RIS, IS
- [ ] Further selection mechanisms
- [x] NSGA2

#Remarks
- Please add some comments

# How to use it?
- Clone the repository:
  ```git clone https://github.com/maxreiss123/VectorizedGeneExpressionProgramming.git```
- Than you can run:
  ```julia --project=. ```
- After the REPL appears, press:
  ```]```
- Within the packet manager enter:
  ```instantiate```


## References
[1] Ferreira, C. (2001). Gene Expression Programming: a New Adaptive Algorithm for Solving Problems. Retrieved from https://arxiv.org/abs/cs/0102027
