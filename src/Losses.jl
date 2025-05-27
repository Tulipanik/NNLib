
module Losses
    using AutoDiffLib
    # using Statistics: mean

    export binarycrossentropy

    const EPS = 1e-7

    mean(x) = sum(x) / length(x)


    # zeralizować na graph nodeach
    function binarycrossentropy(ŷ, y)
        @assert size(ŷ) == size(y)
        eps = EPS
        ŷ_clipped = clamp.(ŷ, eps, 1 - eps)
        to_mean = y .* log.(ŷ_clipped) .+ (1 .- y) .* log.(1 .- ŷ_clipped)
        loss = -mean(to_mean)
        # println(loss)
        return loss
    end

end
