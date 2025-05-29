using Pkg
cd("example")
Pkg.activate("..")
using Revise

using JLD2
X_train = load("data/imdb_dataset_prepared.jld2", "X_train")
y_train = load("data/imdb_dataset_prepared.jld2", "y_train")
X_test = load("data/imdb_dataset_prepared.jld2", "X_test")
y_test = load("data/imdb_dataset_prepared.jld2", "y_test")

# println(typeof(y_train))

using Revise
using NNLib, Printf, Statistics

dataset = DataLoader((X_train, y_train), batchsize=64, shuffle=true)
# println(typeof(dataset[1][2]))
model = Chain([Dense(size(X_train, 1), 32, σ=ReLU), Dense(32, 1, σ=Sigmoid)])

function loss(m, x, y)
    mse_loss(m(x), y)
end
accuracy(m, x, y) = Statistics.mean((m(Variable(x)).value .> 0.5) .== (y .> 0.5))

opt = Adam()
epochs = 5

for epoch in 1:epochs
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    t = @elapsed begin
        for (x, y) in dataset
            grads = gradient(model) do m
                ŷ = Variable(x, "x")
                l = loss(m, ŷ, y)
                total_loss += l.value
                total_acc += accuracy(m, x, y)
                return l
            end
            # grads - gradienty W w warstwach
            for (layer, grad) in zip(model.layers, grads)
                apply!(opt, layer.W.value, grad[1])
                apply!(opt, layer.b.value, grad[2])
            end
            num_samples += 1
        end
    end
    train_loss = total_loss / num_samples
    train_acc = total_acc / num_samples

    test_acc = accuracy(model, X_test, y_test)
    test_loss = loss(model, Variable(X_test), y_test).value

    println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.2f, a: %.2f) \tTest: (l: %.2f, a: %.2f)",
        epoch, t, train_loss, train_acc, test_loss, test_acc))
end