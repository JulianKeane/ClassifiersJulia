using LinearAlgebra
using Statistics
using SparseArrays

#classification methods
classification_rule(X::Matrix,w) = mul!(transpose(w),X) .> 0
SVMclassificiationrule(X::Matrix, w) = 2*classification_rule(X,w).-1
score(X::Matrix, Y::AbstractArray, w) = mean(classification_rule(X,w).=Y)

#initialization methods
initWeightsLike(X::Matrix) = zero(X[1, :])

#analytic methods for logistic regression

log_prob(X::Matrix, Y, w) = 1//(1+exp(-1*Diagonal(Y)*(X*w)))

#computational method methods
function gradientDescent(gradFunction::Function, learning_rate, itr::Integer, X::Matrix, Y::Matrix)
    for i = 1:itr
        w = w-learning_rate*gradFunction(X,Y,w)
    end
end

function SLPFit(X,Y)
    w = zero(X[1,:])
end

using Pkg 
