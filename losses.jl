module LossFunction


export get_loss_function

function r2_score(y_true::Vector{T}, y_pred::Vector{T}) where T<:Real
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

function mean_squared_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:Real

        d::T = zero(T)
        @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
              temp = (y_true[i]-y_pred[i])
              d += temp*temp
        end
        return d/length(y_true)
end
      
function root_mean_squared_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:Real
          d::T = zero(T)
          @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
                temp = (y_true[i]-y_pred[i])
                d += temp*temp
          end
          return abs2(d/length(y_true))
end
    


loss_functions = Dict{String, Function}(
    "r2_score" => r2_score,
    "mse" => mean_squared_error,
    "rmse" => root_mean_squared_error
    )

function get_loss_function(name::String)
    return loss_functions[name]
end


end