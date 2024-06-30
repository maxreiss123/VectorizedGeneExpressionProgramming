using Random
using Statistics
using LinearAlgebra
using ProgressMeter
using OrderedCollections
using DynamicExpressions
using Logging
#include("ErrorFunctions.jl")

Logging.disable_logging(Logging.Info)
Random.seed!(0)


function find_indices_with_sum(arr, target_sum, num_indices)
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


function compile_to_function_string(rek_string, arity_map)
    stack = []
    for elem in reverse(rek_string)
        if get(arity_map, elem, 0) == 2
            op1 = pop!(stack)
            op2 = pop!(stack)
            push!(stack, "($op1$elem$op2)")
        elseif get(arity_map, elem, 0) == 1
            op1 = pop!(stack)
            push!(stack, "$elem($op1)")
        elseif get(arity_map, elem, 0) == -1
            op1 = pop!(stack)
            push!(stack, "($op1)$elem")
        else
            push!(stack, elem)
        end
    end
    return last(stack)
end

function compile_to_cranmer_datatype(rek_string, arity_map, callbacks::Dict, nodes::Dict)
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
    catch
        print(rek_string)
    end

    return last(stack)
end


struct Toolbox
    gene_count::Int
    head_len::Int
    symbols::OrderedDict
    gene_connections::Vector{Int}
    mutation_prob::Float64
    crossover_prob::Float64
    fusion_prob::Float64
    char_to_id::OrderedDict
    id_to_char::OrderedDict
    headsyms::Vector{Int}
    tailsyms::Vector{Int}
    arrity_by_id::OrderedDict
    callbacks::Dict
    nodes::Dict
    
    function Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{String, Int64}, gene_connections::Vector{String}, mutation_prob::Float64, 
        crossover_prob::Float64, fusion_prob::Float64, callbacks::Dict, nodes::Dict)
	    char_to_id = OrderedDict(elem => index for (index, elem) in enumerate(keys(symbols)))
	    id_to_char = OrderedDict(index => elem for (elem, index) in char_to_id)
	    headsyms = [char_to_id[key] for (key, arity) in symbols if arity != 0]
	    tailsyms = [char_to_id[key] for (key, arity) in symbols if arity < 1]
	    arrity_by_id = OrderedDict(index => arity for ((_, arity), index) in zip(symbols, 1:length(symbols)))
	    gene_connections_ids = [char_to_id[elem] for elem in gene_connections]
	    new(gene_count, head_len, symbols, gene_connections_ids, mutation_prob, crossover_prob, fusion_prob, char_to_id, id_to_char, headsyms, tailsyms, arrity_by_id, 
        callbacks, nodes)
     end
end


mutable struct Chromosome
    genes::Vector{Int}
    fitness::Float32
    toolbox::Toolbox
    expression::String
    compiled_function::Any

    function Chromosome(genes::Vector{Int}, toolbox::Toolbox, compile::Bool=false)
        obj = new()
        obj.genes = genes
        obj.fitness = NaN
        obj.toolbox = toolbox
        obj.expression = ""
        if compile
            compile_expression!(obj)
        end
        return obj
    end
end

function compile_expression!(chromosome::Chromosome)
    if chromosome.expression == ""
        expression = _karva_raw(chromosome)
        temp = [chromosome.toolbox.id_to_char[elem] for elem in expression]
        expression_tree = compile_to_cranmer_datatype(temp, chromosome.toolbox.symbols, chromosome.toolbox.callbacks,
        chromosome.toolbox.nodes)
        chromosome.compiled_function = expression_tree
    end
end

function fitness(chromosome::Chromosome)
    return chromosome.fitness
end

function set_fitness!(chromosome::Chromosome, value::Float64)
    chromosome.fitness = value
end

function _karva_raw(chromosome::Chromosome)
    gene_len = chromosome.toolbox.head_len * 2 + 1
    connectionsym = chromosome.genes[1:(chromosome.toolbox.gene_count - 1)]
    genes = chromosome.genes[(chromosome.toolbox.gene_count):end]
    arity_gene_ = map(x -> chromosome.toolbox.arrity_by_id[x], genes)
    rolled_indices = [connectionsym]
    for i in 1:gene_len:length(arity_gene_) - gene_len
        window = arity_gene_[i:i + gene_len]
        window[2:length(window)] .-=1
        indices = find_indices_with_sum(window, 0, 1)
        append!(rolled_indices, [genes[i:i + first(indices)]])
    end
    return vcat(rolled_indices...)
end

function generate_gene(headsyms, tailsyms, headlen)
    head = rand(1:max(maximum(headsyms), maximum(vcat(headsyms, tailsyms))), headlen)
    tail = rand(maximum(headsyms)+1:maximum(tailsyms), 2 * headlen + 1)
    return vcat(head, tail)
end


function generate_chromosome(toolbox::Toolbox)
    connectors = rand(1:maximum(toolbox.gene_connections), toolbox.gene_count - 1)
    genes = vcat([generate_gene(toolbox.headsyms, toolbox.tailsyms, toolbox.head_len) for _ in 1:toolbox.gene_count]...)
    return Chromosome(vcat(connectors, genes), toolbox, true)
end


function generate_population(number, toolbox::Toolbox)
    return [generate_chromosome(toolbox) for _ in 1:number]
end

function create_operator_masks(gene_seq_alpha, gene_seq_beta, pb=0.2)
    alpha_operator = zeros(Int, length(gene_seq_alpha))
    beta_operator = zeros(Int, length(gene_seq_beta))
    indices_alpha = rand(1:length(gene_seq_alpha), min(round(Int,(pb * length(gene_seq_alpha))), length(gene_seq_alpha)))
    indices_beta = rand(1:length(gene_seq_beta), min(round(Int,(pb * length(gene_seq_beta))), length(gene_seq_beta)))
    alpha_operator[indices_alpha] .= 1
    beta_operator[indices_beta] .= 1
    return alpha_operator, beta_operator
end

function create_operator_point_one_masks(gene_seq_alpha, gene_seq_beta, toolbox)
    alpha_operator = zeros(Int, length(gene_seq_alpha))
    beta_operator = zeros(Int, length(gene_seq_beta))
    head_len = toolbox.head_len
    gene_len = head_len * 2 + 1
    
    for i in 0:(toolbox.gene_count - 1)
        ref = i * gene_len + 1
        mid = ref + gene_len ÷ 2  

        point1 = rand(ref:mid)
        point2 = rand((mid + 1):(ref + gene_len - 1))
        alpha_operator[point1:point2] .= 1
        
        point1 = rand(ref:mid)
        point2 = rand((mid + 1):(ref + gene_len - 1))
        beta_operator[point1:point2] .= 1
    end
    
    return alpha_operator, beta_operator
end

function create_operator_point_two_masks(gene_seq_alpha, gene_seq_beta, toolbox)
    alpha_operator = zeros(Int, length(gene_seq_alpha))
    beta_operator = zeros(Int, length(gene_seq_beta))
    head_len = toolbox.head_len
    gene_len = head_len * 2 + 1

    for i in 0:(toolbox.gene_count - 1)
        start = i * gene_len + 1
        quarter = start + gene_len ÷ 4
        half = start + gene_len ÷ 2
        end_gene = start + gene_len - 1


        point1 = rand(start:quarter)
        point2 = rand(quarter+1:half)
        point3 = rand(half+1:end_gene)
        alpha_operator[point1:point2] .= 1
        alpha_operator[point3:end_gene] .= 1


        point1 = rand(start:end_gene)
        point2 = rand(point1:end_gene)
        beta_operator[point1:point2] .= 1
        beta_operator[point2+1:end_gene] .= 1
    end

    return alpha_operator, beta_operator
end


function gene_dominant_fusion(chromosome1::Chromosome, chromosome2::Chromosome, pb=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)
    child_1 = Chromosome(vcat([alpha_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)]...), chromosome1.toolbox)
    child_2 = Chromosome(vcat([beta_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i] for i in 1:length(gene_seq_beta)]...), chromosome1.toolbox)
    return child_1, child_2
end


function gen_rezessiv(chromosome1::Chromosome, chromosome2::Chromosome, pb=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)
    child_1 = Chromosome(vcat([alpha_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)]...), chromosome1.toolbox)
    child_2 = Chromosome(vcat([beta_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i] for i in 1:length(gene_seq_beta)]...), chromosome1.toolbox)
    return child_1, child_2
end

function gene_fussion(chromosome1::Chromosome, chromosome2::Chromosome, pb=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)
    child_1 = Chromosome(vcat([alpha_operator[i] == 1 ? div(gene_seq_alpha[i] + gene_seq_beta[i], 2) : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)]...), chromosome1.toolbox)
    child_2 = Chromosome(vcat([beta_operator[i] == 1 ? div(gene_seq_alpha[i] + gene_seq_beta[i], 2) : gene_seq_beta[i] for i in 1:length(gene_seq_beta)]...), chromosome1.toolbox)
    return child_1, child_2
end

function gene_one_point_cross_over(chromosome1::Chromosome, chromosome2::Chromosome)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_point_one_masks(gene_seq_alpha, gene_seq_beta, chromosome1.toolbox)
    child_1 = Chromosome(vcat([alpha_operator[i] == 1 ? gene_seq_alpha[i] : gene_seq_beta[i] for i in 1:length(gene_seq_alpha)]...), chromosome1.toolbox)
    child_2 = Chromosome(vcat([beta_operator[i] == 1 ? gene_seq_beta[i] : gene_seq_alpha[i] for i in 1:length(gene_seq_beta)]...), chromosome1.toolbox)
    return child_1, child_2
end

function gene_two_point_cross_over(chromosome1::Chromosome, chromosome2::Chromosome)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_point_two_masks(gene_seq_alpha, gene_seq_beta, chromosome1.toolbox)
    child_1 = Chromosome(vcat([alpha_operator[i] == 1 ? gene_seq_alpha[i] : gene_seq_beta[i] for i in 1:length(gene_seq_alpha)]...), chromosome1.toolbox)
    child_2 = Chromosome(vcat([beta_operator[i] == 1 ? gene_seq_beta[i] : gene_seq_alpha[i] for i in 1:length(gene_seq_beta)]...), chromosome1.toolbox)
    return child_1, child_2
end

function gene_mutation(chromosome1::Chromosome, chromosome2::Chromosome, pb=0.2)
    gene_seq_alpha = chromosome1.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_alpha, pb)
    mutation_seq_1  = generate_chromosome(chromosome1.toolbox)
    mutation_seq_2  = generate_chromosome(chromosome2.toolbox)
    child_1 = Chromosome(vcat([alpha_operator[i] == 1 ? mutation_seq_1.genes[i] : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)]...), chromosome1.toolbox)
    child_2 = Chromosome(vcat([beta_operator[i] == 1 ? mutation_seq_2.genes[i] : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)]...), chromosome2.toolbox)
    return child_1, child_2
end



function basic_tournament_selection(population, tournament_size, number_of_winners)
    selected = []
    for _ in 1:number_of_winners
        contenders = rand(population, tournament_size)
        winner = reduce((best, contender) -> contender.fitness < best.fitness ? contender : best, contenders)
        push!(selected, winner)
    end
    return selected
end

function mean_squared_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:Real

  d::Float64 = 0.0
  @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
        temp = (y_true[i]-y_pred[i])
        d += temp*temp
  end
  return d/length(y_true)
end


function compute_fitness(elem, operators, x_data, y_data)
    try    
        if isnan(elem.fitness)
            y_pred = elem.compiled_function(x_data, operators)
            return mean_squared_error(y_data, y_pred)
        else
            return elem.fitness
        end
    catch
        return 10e6
    end
end


function genetic_operations(parent1::Chromosome, parent2::Chromosome, toolbox::Toolbox)
    child1, child2 = parent1, parent2

    if rand() < toolbox.mutation_prob
        child1, child2 = gene_mutation(child1, child2)
    end

    if rand() < toolbox.fusion_prob
        child1, child2 = gene_dominant_fusion(child1, child2)
    end
    if rand() < toolbox.fusion_prob
        child1, child2 = gen_rezessiv(child1, child2)
    end
    if rand() < toolbox.fusion_prob
        child1, child2 = gene_fussion(child1, child2)
    end
    if rand() < toolbox.crossover_prob
        child1, child2 = gene_one_point_cross_over(child1, child2)
    end
    if rand() < toolbox.crossover_prob
        child1, child2 = gene_two_point_cross_over(child1, child2)
    end

    return child1, child2
end

function run_genetic_algorithm(epochs, population_size, gene_count, head_len, symbols, gene_connections, mutation_prob, crossover_prob, fusion_prob)
    #create a function dictionary
    operators =  OperatorEnum(; binary_operators=[+, -, *, /])
    callbacks = Dict(
            "-" => (-),
            "/" => (/),
            "*" => (*),
            "+" => (+)
    )
    nodes = Dict(
        "x_0" => Node(; feature=1),
        "2" => 2,
        "0" => 0
    )
     
    #Generate some data
    x_data = randn(Float32, 1, 20000)
    y_data = x_data.^3 + x_data.^2 + x_data .+ 4
    mating_size = Int(ceil(population_size*0.4))
    
    toolbox = Toolbox(gene_count, head_len, symbols, gene_connections, mutation_prob, crossover_prob, 
    fusion_prob,callbacks, nodes)
    population = generate_population(population_size, toolbox)
    
    @showprogress for epoch in 1:epochs
        Threads.@threads for i in eachindex(population)
            if isnan(population[i].fitness)
                population[i].fitness = compute_fitness(population[i], operators, x_data, y_data)
            end
        end
        

        sort!(population, by = x -> x.fitness)
        if epoch < epochs
            parents = basic_tournament_selection(population, 3, mating_size)

            next_gen = Vector{eltype(population)}(undef, length(parents))
            Threads.@threads for i in 1:2:length(parents) - 1
                parent1 = parents[i]
                parent2 = parents[i + 1]
                #we compile after the last one!
                child1, child2 = genetic_operations(parent1, parent2, toolbox)
                
                compile_expression!(child1)
                compile_expression!(child2)
                
                next_gen[i] = child1
                next_gen[i+1] = child2
            end
            population = vcat(sort(population, by = x -> x.fitness)[1:mating_size], next_gen)
            #print(println(population[1].fitness))
        end
    end
    println(population[1].fitness)
    #Return the winner
    return sort(population, by = x -> x.fitness)[1]
end

#Example Call
#Lessons learned - we need to add a point and a whitespace into the string :D
best_individual = run_genetic_algorithm(1000, 10000, 2, 10, OrderedDict("+" => 2, "*" => 2, "-" => 2, "/" => 2, "x_0" => 0, "2" => 0, "0"=> 0), 
["+", "*"], 0.1, 0.1, 0.1)

