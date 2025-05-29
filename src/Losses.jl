
module Losses
using AutoDiffLib
# using Statistics: mean

export binarycrossentropy, mse_loss

const EPS = 1e-7

# mean(x) = sum(x) / length(x)


# zeralizować na graph nodeach
function binarycrossentropy(ŷ, y)
    # println(y)
    y = y isa Variable ? y : Variable(y, "y")
    # println(typeof(y))
    # print(typeof(ŷ))
    @assert size(ŷ.value) == size(y.value)
    println("siemano")
    @show size((Constant(fill(1, size(y.value))) - y).value)
    @show size((Constant(fill(1, size(ŷ.value))) - ŷ).value)
    eps = Constant(fill(EPS, size(y.value)))
    @show (y .* log(ŷ + eps) + (Constant(fill(1, size(y.value))) - y) .* log(Constant(fill(1, size(ŷ.value))) - ŷ + eps)).value
    to_mean = y .* log(ŷ + eps) + (Constant(fill(1, size(y.value))) - y) .* log(Constant(fill(1, size(ŷ.value))) - ŷ + eps)
    @show size(to_mean.value)
    # @show methods(mean)
    loss = mean(to_mean)
    @show loss.value
    return loss
end

function mse_loss(ŷ, y)
    # Konwertuj y na Variable jeśli nie jest
    y = y isa Variable ? y : Variable(y, "y")
    ŷ = y isa Variable ? ŷ : Variable(y, "ŷ")

    @assert size(ŷ.value) == size(y.value) "Wymiary nie są zgodne: ŷ $(size(ŷ.value)) vs y $(size(y.value))"

    # Oblicz różnicę kwadratową
    # diff = (ŷ - y) .^ Constant(2)

    # @show "w mse loss"

    # @show typeof(y)
    # @show typeof(ŷ)

    # Zwróć średnią
    return mean(Constant(0.5) .* (y - ŷ) .^ Constant(2))
end

end
