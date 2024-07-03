include("vgep.jl")

using .VGEP
using DynamicExpressions
using OrderedCollections


#Example Call
#Define utilized syms as Ordered Dict: Symbol:Arity
utilized_syms = OrderedDict("+" => 2, "*" => 2, "-" => 2, "/" => 2, "x_0" => 0, "2" => 0, "0"=> 0, "x_1" => 0)

#Create connection between genes 
connection_syms = ["+", "*"]

#Define all the elements for the dynamic.jl
operators =  OperatorEnum(; binary_operators=[+, -, *, /])

callbacks = Dict(
        "-" => (-),
        "/" => (/),
        "*" => (*),
        "+" => (+)
)
nodes = OrderedDict(
    "x_0" => Node(; feature=1),
    "x_1" => Node(; feature=2),
    "2" => 2,
    "0" => 0
)
 

#Generate some data
x_data = randn(Float32, 2, 1000)
y_data = @. x_data[1,:] * x_data[1,:] + x_data[1,:] * x_data[2,:] - 2 * x_data[2,:] * x_data[2,:]

#call the function -> return value yields the best:


best=run_GEP(1000,1000,4,10,utilized_syms,operators, callbacks, nodes, x_data,y_data, connection_syms)
@show string(best.compiled_function)
