module VGEPUtils


using OrderedCollections
using DynamicExpressions
using LinearAlgebra
using Optim

export find_indices_with_sum, compile_to_cranmer_datatype, optimize_constants, minmax_scale

    function fast_sqrt_32(x::Real)
        i = reinterpret(UInt32, x)
        i = 0x1fbd1df5 + (i >> 1)
        return reinterpret(Real, i)
    end


    function find_indices_with_sum(arr::Vector{Int}, target_sum::Int, num_indices::Int)
        if arr[1] == -1
            return [1]
        end
        cum_sum = cumsum(arr)
        indices = findall(x -> x == target_sum, cum_sum)
        if length(indices) >= num_indices
        return indices[1:num_indices]
        else
            return [1]
        end
    end

    function compile_to_cranmer_datatype(rek_string::Vector, arity_map::OrderedDict, callbacks::Dict, nodes::OrderedDict)
        stack = []
        try
            for elem in reverse(rek_string)
                if get(arity_map, elem, 0) == 2
                    op1 = (temp = pop!(stack); temp isa String ? nodes[temp] : temp)
                    op2 = (temp = pop!(stack); temp isa String ? nodes[temp] : temp)
                    ops = callbacks[elem]
                    push!(stack, ops(op1,op2))
                elseif get(arity_map, elem, 0) == 1
                    op1 = (temp = pop!(stack); temp isa String ? nodes[temp] : temp)
                    ops = callbacks[elem]
                    push!(stack, ops(op1))
                else
                    push!(stack, elem)
                end
            end
        catch e
            #@error "An error occurred during function compile "
        end
    
        return last(stack)
    end

    function retrieve_constants_from_node(node::AbstractNode)
        constants = Float64[]
        for op in node
            if op isa AbstractNode && op.degree == 0 && op.constant
                push!(constants, convert(Float64, op.val))
            end
        end
        constants
    end
    
    function update_constants!(node::AbstractNode, new_constants::Vector{Float64})
            constant_index = 1
            for op in node
                if op isa AbstractNode && op.degree == 0 && op.constant
                    op.val = new_constants[constant_index]
                    constant_index += 1
                end
            end
    end

    function objective(params, node::AbstractNode, x_data::AbstractArray{T}, y_data::AbstractArray{T}, loss::Function, operators::OperatorEnum) where T<:AbstractFloat
        update_constants!(node, params)
        y_pred = node(x_data, operators)
        return loss(y_pred, y_data)
    end

    function optimize_constants(node::AbstractNode, 
        initial_fitness::Real,
        x_data::AbstractArray{T}, 
        y_data::AbstractArray{T}, loss::Function, operators::OperatorEnum; max_iterations::Int=250) where T<:AbstractFloat
        
        initial_constants = retrieve_constants_from_node(node)
        try
            obj(p) = objective(p, node, x_data, y_data, loss, operators)
            result = optimize(
                obj,
                initial_constants,
                NelderMead(),
                Optim.Options(
                    show_trace = false,
                    iterations = max_iterations,
                    g_tol = 1e-8
                )
            )
            
            optimized_constants = Optim.minimizer(result)
            update_constants!(node, optimized_constants)
            return node, Optim.minimum(result)
        catch 
            return node, initial_fitness
        end
    end



    function _minmax_scale!(X::AbstractMatrix{T}; feature_range=(zero(T), one(T))) where T<:AbstractFloat
        min_vals = minimum(X, dims=1)
        max_vals = maximum(X, dims=1)
        range_width = max_vals .- min_vals
        
        a, b = feature_range
        scale = (b - a) ./ range_width
        
        @inbounds @simd for j in axes(X, 2)
            if range_width[j] ≈ zero(T)
                X[:, j] .= (a + b) / 2
            else
                @simd for i in axes(X, 1)
                    X[i, j] = (X[i, j] - min_vals[j]) * scale[j] + a
                end
            end
        end
        
        return X
    end
    
    function minmax_scale(X::AbstractMatrix{T}; feature_range=(zero(T), one(T))) where T<:AbstractFloat
        return _minmax_scale!(copy(X); feature_range=feature_range)
    end

end