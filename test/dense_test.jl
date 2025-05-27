using NNLib
using AutoDiffLib
using Test

@testset "NNLib.jl" begin
    x = Variable([1.0, 2.0], "x")
    dense = Dense(2, 3, σ=ReLU)

    y=@toposort dense(x)
    # println(y)
    # println(y)

    backward!(y)
    print(x.grad)
end