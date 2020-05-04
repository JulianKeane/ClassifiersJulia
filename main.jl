using LinearAlgebra

classification_rule(x::AbstractMatrix,w) = transpose(w)*x .> 0
SVMclassificiationrule(x::AbstractMatrix, w) = 2*classification_rule(x,w).-1

