using NNLib
using AutoDiffLib
using Test

@testset "NNLib.jl" begin
    x = Variable([1.0, 2.0], "x")
    y_true = Float32[0.5, 1.0, 1.5]

    model = Chain([Dense(2, 3, σ=ReLU)])

    y_pred = model(x)
    loss = mse_loss(y_pred.value, y_true)

    z = @toposort loss

    backward!(z)
end