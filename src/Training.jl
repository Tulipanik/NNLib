module Train

using Printf, Statistics
using ..Gradients: gradient
using ..Optimizer: update!
using ..DataLoaderModule: DataLoader
using AutoDiffLib

export train!

accuracy(m, x, y) = Statistics.mean((m(Variable(x)).value .> 0.5) .== (y .> 0.5))


function train!(model, X_train, y_train, X_test, y_test, loss, opt; accuracy=accuracy, epochs=5)

    println("Starting process of learning...")
    dataset = DataLoader((X_train, y_train), batchsize=64, shuffle=true)

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
                for (layer, grad) in zip(model.layers, grads)
                    update!(opt, layer.W.value, grad[1], copy(layer.W.value))
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
    println("Finishing process of learning")

end

end

