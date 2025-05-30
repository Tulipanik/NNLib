using Test
using NNLib
using AutoDiffLib

@testset "Chain forward pass" begin
    x_data = [1.0, 2.0]
    x = Variable(reshape(x_data, :, 1), "x")

    model = Chain([
        Dense(2, 3, σ=ReLU),
        Dense(3, 1, σ=identity)
    ])

    y = @toposort model(x)

    backward!(y)

    @test length(x.grad) == length(x.value)
end
