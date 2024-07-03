module LossFunction
using Statistics

export get_loss_function


function floor_to_n10p(x::T) where T<:AbstractFloat
    abs_x = abs(x)
    return abs_x > zero(T) ? T(10^floor(log10(abs_x))) : eps(T)
end


function r2_score(y_true::Vector{T}, y_pred::Vector{T}) where T<:AbstractFloat
    len_y = length(y_true)
    y_mean = mean(y_true)
    

    ss_total::T = zero(T)
    @inbounds @simd for i in 1:len_y
	temp = y_true[i]-y_mean
        ss_total += temp*temp
    end

    ss_residual::T = zero(T)
    @inbounds @simd for i in 1:len_y
	temp = y_true[i] - y_pred[i]
	ss_residual += temp*temp
    end
    
    r2 = 1.0 - (ss_residual / ss_total)
    
    return r2
end

function r2_score_floor(y_true::Vector{T}, y_pred::Vector{T}) where T<:AbstractFloat
    max_abs_value = maximum(abs, vcat(y_true, y_pred))

    if max_abs_value == zero(T)
        return one(T)
    end
    
    scale_factor = T(10^floor(log10(max_abs_value)))
    
    # Scale both y_true and y_pred
    y_true_scaled = y_true ./ scale_factor
    y_pred_scaled = y_pred ./ scale_factor
    
    return r2_score(y_true_scaled, y_pred_scaled)
end



function mean_squared_error(y_true::Vector{T}, y_pred::Vector{T}) where T<:AbstractFloat
        d::T = zero(T)
        @assert length(y_true) == length(y_pred)
        @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
              temp = (y_true[i]-y_pred[i])
              d += temp*temp
        end
        return d/length(y_true)
end
      
function root_mean_squared_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:AbstractFloat
          d::T = zero(T)
          @assert length(y_true) == length(y_pred)
          @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
                temp = (y_true[i]-y_pred[i])
                d += temp*temp
          end
          return abs2(d/length(y_true))
end

function mean_absolute_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:AbstractFloat
    d::T = zero(T)
    @assert length(y_true) == length(y_pred)
    @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
        d += abs(y_true[i]-y_pred[i])
    end
    return d/length(y_true)
end

function save_root_mean_squared_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:AbstractFloat
    @assert length(y_true) == length(y_pred)
    
    max_abs_value = maximum(abs, vcat(y_true, y_pred))
    
    if max_abs_value == zero(T)
        return zero(T)
    end
    
    scale_factor = T(10^floor(log10(max_abs_value)))
    
    d::T = zero(T)
    @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
        y_true_scaled = y_true[i] / scale_factor
        y_pred_scaled = y_pred[i] / scale_factor
        temp = (y_true_scaled - y_pred_scaled) / (abs(y_true_scaled) + eps(T))
        d += temp * temp
    end
    
    return sqrt(d / length(y_true))
end



loss_functions = Dict{String, Function}(
    "r2_score" => r2_score,
    "r2_score_f" => r2_score_floor,
    "mse" => mean_squared_error,
    "rmse" => root_mean_squared_error,
    "mae" => mean_absolute_error,
    "srsme" => save_root_mean_squared_error
    )

function get_loss_function(name::String)
    return loss_functions[name]
end


end