using NNLib
using AutoDiffLib
using Test

@testset "NNLib.jl" begin
    model = Chain([
    Dense(2, 4, σ=ReLU),
    Dense(4, 1, σ=identity)
    ])

    loss_fn(ŷ, y) = mse_loss(ŷ, y)

    opt = Adam()

    data = [([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([0.0, 0.0], [0.0])]

    train!(model, loss_fn, data, opt, epochs=100)
end