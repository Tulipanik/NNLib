
module Losses
using AutoDiffLib

export binarycrossentropy, mse_loss

const EPS = 1e-7

function binarycrossentropy(ŷ, y)
    y = y isa Variable ? y : Variable(y, "y")
    @assert size(ŷ.value) == size(y.value)
    eps = Constant(fill(EPS, size(y.value)))
    to_mean = y .* log(ŷ + eps) + (Constant(fill(1, size(y.value))) - y) .* log(Constant(fill(1, size(ŷ.value))) - ŷ + eps)
    loss = mean(to_mean)
    @show loss.value
    return loss
end

function mse_loss(ŷ, y)
    y = y isa Node ? y : Variable(y, "y")
    ŷ = ŷ isa Node ? ŷ : Variable(ŷ, "ŷ")

    @assert size(ŷ.value) == size(y.value) "Wymiary nie są zgodne: ŷ $(size(ŷ.value)) vs y $(size(y.value))"
    return mean(Constant(0.5) .* (y - ŷ) .^ Constant(2))
end

end
