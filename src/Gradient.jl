module Gradients

using AutoDiffLib

export gradient

function gradient(f, model)
    loss = f(model)
    # print(typeof(loss))
    # @show size(loss.value)

    z = topological_sort(loss)

    backward!(z)

    grads = [(layer.W.grad, layer.b.grad) for layer in model.layers]

    return grads
end

function gradient(f::Function)
    return model -> gradient(f, model)
end

end