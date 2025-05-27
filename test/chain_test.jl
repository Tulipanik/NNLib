using Test
using NNLib
using AutoDiffLib

@testset "Chain forward pass" begin
    x_data = [1.0, 2.0]
    x = Variable(reshape(x_data, :, 1), "x")  # 2×1 input

    model = Chain([
        Dense(2, 3, σ=ReLU),
        Dense(3, 1, σ=identity)
    ])

    y = @toposort model(x)

    # @show y

    # @test y isa Variable
    # @test size(y.value) == (1, 1)

    backward!(y)

    @test length(x.grad) == length(x.value)
end
