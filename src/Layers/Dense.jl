module DenseLayer

using AutoDiffLib

export Dense

mutable struct Dense
    W::Variable
    b::Variable
    σ::Function
end

# xavier initialization

function Dense(in_features::Int, out_features::Int; σ=identity)
    W = Variable(randn(out_features, in_features) * sqrt(2 / in_features), "W")
    b = Variable(zeros(out_features), "b")
    return Dense(W, b, σ)
end

function (layer::Dense)(x::Node)
    return layer.σ(layer.W * x + layer.b)
end

end
