module Model

using AutoDiffLib
using ..DenseLayer

export Chain

struct Chain
    layers::Vector{Any}
end

function (nn::Chain)(x::Variable)
    for layer in nn.layers
        x = layer(x)
    end
    return x
end

end
