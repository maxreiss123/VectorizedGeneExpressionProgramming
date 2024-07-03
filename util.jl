module VGEPUtils

using OrderedCollections

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

function compile_to_cranmer_datatype(rek_string::Vector, arity_map::OrderedDict, callbacks::Dict, nodes::Dict)
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


end