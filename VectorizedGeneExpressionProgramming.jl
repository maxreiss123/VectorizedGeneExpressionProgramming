using Random
using Statistics
using Plots
using LinearAlgebra
using ProgressMeter
using OrderedCollections
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

function mean_absolute_error(y_true, y_pred)
    return sum(abs.(y_pred-y_true))/length(y_true)
end

function mean_squared_error(y_true, y_pred)
    return sum((y_true.-y_pred).^2)/length(y_true)
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
    data_vars::Dict
    
    function Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{String, Int64}, gene_connections::Vector{String}, mutation_prob::Float64, 
        crossover_prob::Float64, fusion_prob::Float64, data_vars::Dict)
	    char_to_id = OrderedDict(elem => index for (index, elem) in enumerate(keys(symbols)))
	    id_to_char = OrderedDict(index => elem for (elem, index) in char_to_id)
	    headsyms = [char_to_id[key] for (key, arity) in symbols if arity != 0]
	    tailsyms = [char_to_id[key] for (key, arity) in symbols if arity < 1]
	    arrity_by_id = OrderedDict(index => arity for ((_, arity), index) in zip(symbols, 1:length(symbols)))
	    gene_connections_ids = [char_to_id[elem] for elem in gene_connections]
	    new(gene_count, head_len, symbols, gene_connections_ids, mutation_prob, crossover_prob, fusion_prob, char_to_id, id_to_char, headsyms, tailsyms, arrity_by_id, data_vars)
     end
end


mutable struct Chromosome
    genes::Vector{Int}
    fitness::Float64
    toolbox::Toolbox
    expression::String
    compiled_function::Function

    function Chromosome(genes::Vector{Int}, toolbox::Toolbox)
        obj = new()
        obj.genes = genes
        obj.fitness = NaN
        obj.toolbox = toolbox
        obj.expression = ""
        obj.compiled_function = function() end
        compile_expression!(obj,toolbox.data_vars)
        return obj
    end
end

function compile_expression!(chromosome::Chromosome, data_vars::Dict)
    if chromosome.expression == ""
        expression = _karva_raw(chromosome)
        temp = [chromosome.toolbox.id_to_char[elem] for elem in expression]
        expression_string = compile_to_function_string(temp, chromosome.toolbox.symbols)

        # Create a function string with all data variables as arguments
        args = join(["$(var) = $(data_vars[var])" for var in keys(data_vars)], ", ")
        function_string = "($args) -> $expression_string"

        # Compile and store the function
        chromosome.compiled_function = eval(Meta.parse(function_string))
        chromosome.expression = expression_string
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
    arity_gene_[2:end] .-= 1
    rolled_indices = [connectionsym]
    for i in 1:gene_len:length(arity_gene_) - gene_len
        window = arity_gene_[i:i + gene_len-1]
        indices = find_indices_with_sum(window, 0, 1)
        append!(rolled_indices, [genes[i:i + first(indices)]])
    end
    return vcat(rolled_indices...)
end

#Side note: the bang operator marks a change of the object
#function compile_expression!(chromosome::Chromosome)
#    if chromosome.expression == ""
#        expression = _karva_raw(chromosome)
#        temp = [chromosome.toolbox.id_to_char[elem] for elem in expression]
#        chromosome.expression = compile_to_function_string(temp, chromosome.toolbox.symbols)
#    end
#end

function generate_gene(headsyms, tailsyms, headlen)
    head = rand(1:max(maximum(headsyms), maximum(vcat(headsyms, tailsyms))), headlen)
    tail = rand(maximum(headsyms)+1:maximum(tailsyms), 2 * headlen + 1)
    return vcat(head, tail)
end


function generate_chromosome(toolbox::Toolbox)
    connectors = rand(1:maximum(toolbox.gene_connections), toolbox.gene_count - 1)
    genes = vcat([generate_gene(toolbox.headsyms, toolbox.tailsyms, toolbox.head_len) for _ in 1:toolbox.gene_count]...)
    return Chromosome(vcat(connectors, genes), toolbox)
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

function gene_dominant_fusion(chromosome1::Chromosome, chromosome2::Chromosome, pb=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)
    child_1 = Chromosome([alpha_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)], chromosome1.toolbox)
    child_2 = Chromosome([beta_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i] for i in 1:length(gene_seq_beta)], chromosome1.toolbox)
    return child_1, child_2
end


function gen_rezessiv(chromosome1::Chromosome, chromosome2::Chromosome, pb=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)
    child_1 = Chromosome([alpha_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)], chromosome1.toolbox)
    child_2 = Chromosome([beta_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i] for i in 1:length(gene_seq_beta)], chromosome1.toolbox)
    return child_1, child_2
end

function gene_fussion(chromosome1::Chromosome, chromosome2::Chromosome, pb=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)
    child_1 = Chromosome([alpha_operator[i] == 1 ? div(gene_seq_alpha[i] + gene_seq_beta[i], 2) : gene_seq_alpha[i] for i in 1:length(gene_seq_alpha)], chromosome1.toolbox)
    child_2 = Chromosome([beta_operator[i] == 1 ? div(gene_seq_alpha[i] + gene_seq_beta[i], 2) : gene_seq_beta[i] for i in 1:length(gene_seq_beta)], chromosome1.toolbox)
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


function run_genetic_algorithm(epochs, population_size, gene_count, head_len, symbols, gene_connections, mutation_prob, crossover_prob, fusion_prob)
    #Generate some data
    x_data = range(-1, stop=1, length=100)
    y_data = x_data.^3 + x_data.^2 + x_data .+ 4
    data_set = Dict("x_0" => x_data)
    for (var_name, value) in data_set
        eval(Meta.parse("$var_name = $value"))
    end
    # Initialize toolbox and population
    toolbox = Toolbox(gene_count, head_len, symbols, gene_connections, mutation_prob, crossover_prob, fusion_prob,data_set)
    population = generate_population(population_size, toolbox)
    #Loop over the epochs 
    @showprogress for epoch in 1:epochs
        for elem in population
            try
                #println(elem.expression)
                y_pred = elem.compiled_function()
                elem.fitness = mean_squared_error(y_data, y_pred)
            catch
                elem.fitness = 10e6
            end
        end
        sort!(population, by = x -> x.fitness)
        println(population[1].fitness)
        if epoch < epochs
            parents = basic_tournament_selection(population, 3, length(population) ÷ 2)

            next_gen = []
            for i in 1:2:length(parents) - 1
                parent1 = parents[i]
                parent2 = parents[i + 1]
                child1, child2 = gene_dominant_fusion(parent1, parent2)
                child1, child2 = gen_rezessiv(child1, child2)
                child1, child2 = gene_fussion(child1, child2)
                push!(next_gen, child1)
                push!(next_gen, child2)
            end

            population = vcat(sort(population, by = x -> x.fitness)[1:length(population) ÷ 2], next_gen)
        end
    end
    println(population[1].fitness)
    #Return the winner
    return sort(population, by = x -> x.fitness)[1]
end

#Example Call
#Lessons learned - we need to add a point and a whitespace into the string :D
best_individual = run_genetic_algorithm(100, 1000, 2, 10, OrderedDict(" .+ " => 2, " .* " => 2, " .- " => 2, " ./ " => 2, "x_0" => 0, "2" => 0), [" .+ ", " .- "], 0.1, 0.1, 0.1)

