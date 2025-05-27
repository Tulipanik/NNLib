using Pkg
cd("example")
Pkg.activate("..")

using JLD2
X_train = load("data/imdb_dataset_prepared.jld2", "X_train")
y_train = load("data/imdb_dataset_prepared.jld2", "y_train")
X_test = load("data/imdb_dataset_prepared.jld2", "X_test")
y_test = load("data/imdb_dataset_prepared.jld2", "y_test")

using Revise
using NNLib, Printf, Statistics

dataset = DataLoader((X_train, y_train), batchsize=64, shuffle=true)
model = Chain([Dense(size(X_train, 1), 32, σ=ReLU), Dense(32, 1, σ=ReLU)])

function loss(x, y)
    # println("eloelo")
    binarycrossentropy(x, y)
end
accuracy(m, x, y) = mean((m(x) .> 0.5) .== (y .> 0.5))

opt = Adam()
epochs = 5

for epoch in 1:epochs
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    t = @elapsed begin
        for (x,y) in dataset
            grads = gradient(model) do m
                ŷ = m(Variable(x, "x"))
                l = loss(ŷ.value, y)
                total_loss += l
                return l
            end
            for (layer, grad) in zip(model.layers, grads)
                apply!(opt, layer, grad)
            end
            num_samples += 1
        end
    end
    train_loss = total_loss / num_samples
    train_acc = total_acc / num_samples

    test_acc = accuracy(model, X_test, y_test)
    test_loss = loss(X_test, y_test)

    println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.2f, a: %.2f) \tTest: (l: %.2f, a: %.2f)", 
    epoch, t, train_loss, train_acc, test_loss, test_acc))
end