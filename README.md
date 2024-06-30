# VectorizedGeneExpressionProgramming
- This is a symbolic regression algorithm with Gene Expression Programming as a role model. (the operators changed - for experimental reasons)
- At a certain amount of expression, the list iterations within the genetic operators produce a lot of computational overhead
- Here, we project these into a vector space and use simple vector operations
- Velocity check: 200.000 datapoints, 1000 candidate solutions, mating size 0.4, four threads
- - Julia: 69s
  - Python: 

# Todo 
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

