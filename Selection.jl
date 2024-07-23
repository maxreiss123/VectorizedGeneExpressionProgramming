module EvoSelection 

export selection_NSGA

function dominates_(a::Tuple, b::Tuple)
    return all(a.<=b) && any(a .< b)
end


function fast_non_dominated_sort(population::Vector{Tuple}) 
    pop_size = length(population)
    dom_counts = zeros(Int, n)
    dom_lists = [Int[] for _ in  1:n]
    fronts = [Int[]]

    @inbounds for i in 1:pop_size
        for j in (i+1):pop_size
            if dominates_(population[i], population[j])
                push!(dom_lists[i],j)
                dom_counts[j] +=1
            elseif dominates_(population[j], population[i])
                push!(dom_lists[j], i)
                dom_counts[i] +=1
            end
        end
        if dom_counts[i] == 0
            push!(fronts[1],i)
        end
    end

    front_idx = 1
    while !isempty(fronts[front_idx])
        next_front = Int[]
        for i  in fronts[front_idx]
            for j in dom_lists[i]
                dom_counts[j] -= 1
                if dom_counts[j] == 0
                    push!(next_front, j)
                end
            end
        end
        front_idx +=1
        push!(fronts, next_front)
    end

    pop!(fronts)
    return fronts
end

function assign_crowding_distance(front::Vector{Int}, population::Vector{Tuple})
    n = length(front)
    objectives_count = length(first(population))
    
    distances = zeros(Float64, n)

    min_objectives = fill(Inf, objectives_count)
    max_objectives = fill(-Inf, objectives_count)
    for i in front
        for m in 1:objectives_count
            min_objectives[m] = min(min_objectives[m], population[i][m])
            max_objectives[m] = max(max_objectives[m], population[i][m])
        end
    end

    @inbounds for m in 1:objectives_count
        sorted_indices = sort(front, by = i -> population[i][m])
        distances[sorted_indices[1]] = distances[sorted_indices[end]] = Inf

        scale = max_objectives[m] - min_objectives[m]
        if scale > 0
            for i in 2:n-1
                normalized_diff = (population[sorted_indices[i+1]][m] - population[sorted_indices[i-1]][m]) / scale
                distances[sorted_indices[i]] += normalized_diff
            end
        end
    end
    return distances
end

function selection_NSGA(population::Vector{Tuple}, num_to_select::Int)
    fronts = fast_non_dominated_sort(population)
    n_fronts = length(fronts)
    
    selected_indices = Vector{Int}(undef, num_to_select)
    selected_count = 0
    pareto_fronts = Dict{Int, Vector{Int}}()
    
    @inbounds for front_idx in 1:n_fronts
        front = fronts[front_idx]
        pareto_fronts[front_idx] = front
        front_size = length(front)
        
        if selected_count + front_size <= num_to_select
            copyto!(selected_indices, selected_count + 1, front, 1, front_size)
            selected_count += front_size
        else
            remaining_slots = num_to_select - selected_count
            crowding_distances = assign_crowding_distance(front, population)
            
            partialsort!(front, 1:remaining_slots, by = i -> crowding_distances[i], rev=true)
            
            copyto!(selected_indices, selected_count + 1, front, 1, remaining_slots)
            selected_count = num_to_select
            break
        end
        
        if selected_count == num_to_select
            break
        end
    end
    
    return resize!(selected_indices, selected_count), pareto_fronts
end


end
