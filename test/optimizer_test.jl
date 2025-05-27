using NNLib
using AutoDiffLib
using Test

@testset "NNLib.jl" begin
    opt = Adam()

    params = [rand(3, 3), rand(3)]

    grads = [randn(3, 3), randn(3)]

    for (x, grad) in zip(params, grads)
        Δ = apply!(opt, x, grad)
        x .-= Δ
    end
end