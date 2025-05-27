module Gradients

using AutoDiffLib

export gradient

function gradient(f, model)
    loss = f(model)
    print(typeof(loss))

    z = topological_sort(loss)

    println(typeof(z))

    backward!(z)

    grads = [layer.grads for layer in model.layers]

    return grads
end

function gradient(f::Function)
    return model -> gradient(f, model)
end

end