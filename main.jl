using LinearAlgebra
using Statistics

#classification methods
classification_rule(X::AbstractMatrix,w) = transpose(w)*x .> 0
SVMclassificiationrule(X::AbstractMatrix, w) = 2*classification_rule(X,w).-1
score(X::AbstractMatrix, Y::AbstractArray, w) = mean(classification_rule(X,w).=Y)

#initialization methods
initWeightsLike(X::AbstractMatrix) = zero(X[1, :])

#computational method methods
function gradientDescent(gradFunction::Function, learning_rate, itr::Integer, X::AbstractMatrix, Y::AbstractMatrix)
    for i = 1:itr
        w = w-learning_rate*gradFunction(X,Y,w)
    end
end