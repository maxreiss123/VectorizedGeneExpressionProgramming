module VGEP 

include("losses.jl")
include("util.jl")

using .LossFunction
using .VGEPUtils


using Random
using Statistics
using LinearAlgebra
using ProgressMeter
using OrderedCollections
using DynamicExpressions
using Logging



export run_GEP


#=
**** Base Types*******
=#

struct Toolbox
    gene_count::Int
    head_len::Int
    symbols::OrderedDict
    gene_connections::Vector{Int}
    mutation_prob::Real
    crossover_prob::Real
    fusion_prob::Real
    char_to_id::OrderedDict
    id_to_char::OrderedDict
    headsyms::Vector{Int}
    tailsyms::Vector{Int}
    arrity_by_id::OrderedDict
    callbacks::Dict
    nodes::OrderedDict
    
    function Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{String, Int64}, gene_connections::Vector{String}, mutation_prob::Real, 
        crossover_prob::Real, fusion_prob::Real, callbacks::Dict, nodes::OrderedDict)
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
    fitness::AbstractFloat
    toolbox::Toolbox
    compiled_function::Any
    compiled::Bool
    fitness_r2::AbstractFloat

    function Chromosome(genes::Vector{Int}, toolbox::Toolbox, compile::Bool=false)
        obj = new()
        obj.genes = genes
        obj.fitness = NaN
        obj.toolbox = toolbox
	    obj.fitness_r2 = 0.0
        if compile 
            compile_expression!(obj)
            obj.compiled = true
        end
        return obj
    end
end

function compile_expression!(chromosome::Chromosome)
    if !chromosome.compiled
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

function set_fitness!(chromosome::Chromosome, value::AbstractFloat)
    chromosome.fitness = value
end

function _karva_raw(chromosome::Chromosome)
    gene_len = chromosome.toolbox.head_len * 2 + 1
    connectionsym = chromosome.genes[1:(chromosome.toolbox.gene_count - 1)]
    genes = chromosome.genes[(chromosome.toolbox.gene_count):end]
    arity_gene_ = map(x -> chromosome.toolbox.arrity_by_id[x], genes)
    rolled_indices = [connectionsym]
    for i in 1:(gene_len-1):length(arity_gene_)-gene_len
        window = arity_gene_[i:i + gene_len]
        window[2:length(window)] .-=1
        indices = find_indices_with_sum(window, 0, 1)
        append!(rolled_indices, [genes[i:i + first(indices)-1]])
    end
    return vcat(rolled_indices...)
end


#=
**** Genetic operators*******
=#

function generate_gene(headsyms::Vector{Int}, tailsyms::Vector{Int}, headlen::Int)
    head = rand(vcat(headsyms,tailsyms), headlen)
    tail = rand(tailsyms, headlen + 1)
    return vcat(head, tail)
end


function generate_chromosome(toolbox::Toolbox)
    connectors = rand(1:maximum(toolbox.gene_connections), toolbox.gene_count - 1)
    genes = vcat([generate_gene(toolbox.headsyms, toolbox.tailsyms, toolbox.head_len) for _ in 1:toolbox.gene_count]...)
    return Chromosome(vcat(connectors, genes), toolbox, true)
end


function generate_population(number::Int, toolbox::Toolbox)
  population = Vector{Chromosome}(undef,number)
  Threads.@threads for i in 1:number
        @inbounds population[i] = generate_chromosome(toolbox)
  end
  return population
end


function create_operator_masks(gene_seq_alpha::Vector{Int}, gene_seq_beta::Vector{Int}, pb::Real=0.2)
    alpha_operator = zeros(Int, length(gene_seq_alpha))
    beta_operator = zeros(Int, length(gene_seq_beta))
    indices_alpha = rand(1:length(gene_seq_alpha), min(round(Int,(pb * length(gene_seq_alpha))), length(gene_seq_alpha)))
    indices_beta = rand(1:length(gene_seq_beta), min(round(Int,(pb * length(gene_seq_beta))), length(gene_seq_beta)))
    alpha_operator[indices_alpha] .= 1
    beta_operator[indices_beta] .= 1
    return alpha_operator, beta_operator
end



function create_operator_point_one_masks(gene_seq_alpha::Vector{Int}, gene_seq_beta::Vector{Int}, toolbox::Toolbox)
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


#=
**** Genetic operators*******
=#

function create_operator_point_two_masks(gene_seq_alpha::Vector{Int}, gene_seq_beta::Vector{Int}, toolbox::Toolbox)
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


function gene_dominant_fusion(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)
    child_1 = Chromosome(vcat([alpha_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)]...), chromosome1.toolbox)
    child_2 = Chromosome(vcat([beta_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i] for i in 1:length(gene_seq_beta)]...), chromosome1.toolbox)
    return child_1, child_2
end


function gen_rezessiv(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)
    child_1 = Chromosome(vcat([alpha_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)]...), chromosome1.toolbox)
    child_2 = Chromosome(vcat([beta_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i] for i in 1:length(gene_seq_beta)]...), chromosome1.toolbox)
    return child_1, child_2
end

function gene_fussion(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
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

function gene_mutation(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    gene_seq_alpha = chromosome1.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_alpha, pb)
    mutation_seq_1  = generate_chromosome(chromosome1.toolbox)
    mutation_seq_2  = generate_chromosome(chromosome2.toolbox)
    child_1 = Chromosome(vcat([alpha_operator[i] == 1 ? mutation_seq_1.genes[i] : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)]...), chromosome1.toolbox)
    child_2 = Chromosome(vcat([beta_operator[i] == 1 ? mutation_seq_2.genes[i] : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)]...), chromosome2.toolbox)
    return child_1, child_2
end


function basic_tournament_selection(population::Vector{Chromosome}, tournament_size::Int, number_of_winners::Int)
    selected = []
    for _ in 1:number_of_winners
        contenders = rand(population, tournament_size)
        winner = reduce((best, contender) -> contender.fitness < best.fitness ? contender : best, contenders)
        push!(selected, winner)
    end
    return selected
end


function compute_fitness(elem::Chromosome, operators::OperatorEnum, x_data::AbstractArray{T}, y_data::AbstractArray{T}, loss_function::Function, 
    crash_value::Real; validate::Bool=false) where T<:AbstractFloat
    try    
        if isnan(elem.fitness) || validate
	        y_pred = elem.compiled_function(x_data, operators)
	        return loss_function(y_data, y_pred)
        else
            return elem.fitness
        end
    catch e
        return crash_value
    end
end


function genetic_operations(parent1::Chromosome, parent2::Chromosome, toolbox::Toolbox)
    child1, child2 = parent1, parent2

    if rand() < toolbox.crossover_prob*1.5
        child1, child2 = gene_one_point_cross_over(child1, child2)
    end

    if rand() < toolbox.crossover_prob
        child1, child2 = gene_two_point_cross_over(child1, child2)
    end

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

    return child1, child2
    end

function run_GEP(epochs::Int, 
    population_size::Int, 
    gene_count::Int, 
    head_len::Int, 
    symbols::OrderedDict,
    operators::OperatorEnum,
    callbacks::Dict,
    nodes::OrderedDict,
    x_data::AbstractArray{T},
    y_data::AbstractArray{T},
    gene_connections::Vector{String};
    seed::Int=0,
    loss_fun_str::String="mae", 
    mutation_prob::Real=1.0, 
    crossover_prob::Real=0.3, 
    fusion_prob::Real=0.1,
    mating_::Real=0.7,
    epsilon::Real=0.0) where T<:AbstractFloat

    loss_fun::Function = get_loss_function(loss_fun_str)

    Random.seed!(seed)
    mating_size = Int(ceil(population_size*mating_))
    toolbox = Toolbox(gene_count, head_len, symbols, gene_connections, mutation_prob, crossover_prob, 
    fusion_prob,callbacks, nodes)
    population = generate_population(population_size, toolbox)
    prev_best = -1

    @showprogress for epoch in 1:epochs
        
        Threads.@threads for i in eachindex(population)
            if isnan(population[i].fitness)
                population[i].fitness = compute_fitness(population[i], operators, x_data, y_data, loss_fun, 1e32)
            end
        end

        sort!(population, by = x -> x.fitness)
        
        if (prev_best==-1 || prev_best>population[1].fitness) && epoch % 20 == 0
            eqn, result = optimize_constants(population[1].compiled_function,population[1].fitness ,
            x_data, y_data, get_loss_function("srsme"), operators)
                population[1].fitness = result
                population[1].compiled_function = eqn
                prev_best = result
        end

        if epoch < epochs
            parents = basic_tournament_selection(population, 3, mating_size)

            next_gen = Vector{eltype(population)}(undef, length(parents))
            Threads.@threads for i in 1:2:length(parents) - 1
                parent1 = parents[i]
                parent2 = parents[i + 1]
                child1, child2 = genetic_operations(parent1, parent2, toolbox)
                
                compile_expression!(child1)
                compile_expression!(child2)
                
                next_gen[i] = child1
                next_gen[i+1] = child2
            end
            population = vcat(population[1:(population_size-mating_size-1)], next_gen)
        if population[1].fitness<epsilon
            break
        end
        
        end
    end
    best = sort(population, by = x -> x.fitness)[1]
    best.fitness_r2 = compute_fitness(best,operators, x_data, y_data, get_loss_function("r2_score_f"), 0.0; validate=true)
    return best 
    end

end
