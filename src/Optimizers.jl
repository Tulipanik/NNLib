
module Optimizer
export Adam, apply!

const EPS = 1e-8

abstract type AbstractOptimizer end
mutable struct Adam <: AbstractOptimizer
    eta::Float64
    beta::Tuple{Float64,Float64}
    epsilon::Float64
    state::IdDict{Any,Any}
end

Adam(η::Real=0.001, β::Tuple=(0.9, 0.999), ϵ::Real=EPS) = Adam(η, β, ϵ, IdDict())
Adam(η::Real, β::Tuple, state::IdDict) = Adam(η, β, EPS, state)

# function apply!(o::Adam, x, Δ)
# η, β = o.eta, o.beta

# mt, vt, βp = get!(o.state, x.W.value) do
#     (zero(x.W.value), zero(x.W.value), Float64[β[1], β[2]])
# end :: Tuple{typeof(x.W.value),typeof(x.W.value),Vector{Float64}}

# @. mt = β[1] * mt + (1 - β[1]) * Δ
# @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
# @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η

# βp .= βp .* β

# @show "siema"
# @show o.eta
# @show o.beta
# @show o.epsilon
# # @show o.state

# return Δ
# end

# function apply!(o::Adam, x, Δ)
#     η, β = o.eta, o.beta

#     mt, vt, βp = get!(o.state, x.W.value) do
#         (zero(x.W.value), zero(x.W.value), Float64[β[1], β[2]])
#     end::Tuple{typeof(x.W.value),typeof(x.W.value),Vector{Float64}}

#     @. mt = β[1] * mt + (1 - β[1]) * Δ
#     @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)

#     m̂ = mt / (1 - βp[1])
#     v̂ = vt / (1 - βp[2])

#     @. x.W.value -= η * m̂ / (√(v̂) + o.epsilon)
#     βp .= βp .* β

#     return nothing
# end

function apply!(o::Adam, param::AbstractArray, Δ::AbstractArray)
    η, β = o.eta, o.beta # Learning rate and beta values from the optimizer

    # Retrieve or initialize the state (mt, vt, βp) for the current parameter.
    # `param` itself is used as the key in the IdDict to uniquely identify its state.
    mt, vt, βp = get!(o.state, param) do
        # Initialize mt (first moment estimate) and vt (second moment estimate) with zeros
        # of the same type and shape as the parameter.
        # Initialize βp (bias correction factors) as a mutable array.
        (zero(param), zero(param), Float64[β[1], β[2]])
    end::Tuple{typeof(param),typeof(param),Vector{Float64}}

    # Update biased first moment estimate (mt)
    # mt = β1 * mt + (1 - β1) * Δ
    @. mt = β[1] * mt + (1 - β[1]) * Δ

    # Update biased second raw moment estimate (vt)
    # vt = β2 * vt + (1 - β2) * Δ^2
    @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ) # conj(Δ) handles complex gradients if applicable

    # Compute bias-corrected first moment estimate (m̂)
    # m̂ = mt / (1 - β1^t)
    m̂ = mt / (1 - βp[1])

    # Compute bias-corrected second raw moment estimate (v̂)
    # v̂ = vt / (1 - β2^t)
    v̂ = vt / (1 - βp[2])

    # Update the parameter in-place
    # param = param - η * m̂ / (sqrt(v̂) + epsilon)
    @. param -= η * m̂ / (√(v̂) + o.epsilon)

    # Update the bias correction factors for the next iteration (β1^t and β2^t)
    # βp[1] will become β1^(t+1) and βp[2] will become β2^(t+1)
    βp .= βp .* β

    return nothing # The function modifies `param` in-place, so no return value is needed
end

end