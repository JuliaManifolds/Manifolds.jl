using JET, Manifolds, Test
# load every package Manifolds has an extension for, so that JET analyses the extensions as well
using BoundaryValueDiffEq, Colors, DiffEqCallbacks, Distributions, HybridArrays, NLsolve
using OrdinaryDiffEq, RecipesBase, RecursiveArrayTools

@testset "JET.jl" begin
    # only on released Julia versions, JET follows the compiler closely
    if isempty(VERSION.prerelease)
        JET.test_package(Manifolds; target_modules = (Manifolds,), mode = :typo)
    end
end
