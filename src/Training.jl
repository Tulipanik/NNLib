module Train

using AutoDiffLib

export train!

function train!(model, loss_fn, data, opt; epochs = 10)
    for epoch in 1:epochs
        total_loss = 0.0
        for (x, y) in data
            x = Variable(x, "x")
            ŷ = model(x)

            loss = loss_fn(ŷ.value, y)

            @show loss.value
            total_loss += loss.value

            @show total_loss

            z = @toposort loss
            backward!(z)

            for layer in model.layers
                for (param, grad) in zip(layer.params, layer.grads)
                    apply!(opt, param, grad)
                end
            end
        end
        println("Epoch $epoch, Loss: $total_loss")
    end
end

end

