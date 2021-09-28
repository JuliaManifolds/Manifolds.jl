include("utils.jl")

struct TestSphere{N} <: AbstractManifold{ℝ} end
struct TestSphericalMetric <: AbstractMetric end

using FiniteDifferences, ForwardDiff
using LinearAlgebra: I
import Manifolds: retract!
import ManifoldsBase: manifold_dimension, default_retraction_method

function default_retraction_method(::MetricManifold{ℝ,<:TestSphere,<:TestSphericalMetric})
    return ProjectionRetraction()
end
manifold_dimension(::TestSphere{n}) where {n} = n
function Manifolds.retract!(
    ::MetricManifold{ℝ,TestSphere{n},<:TestSphericalMetric},
    q,
    p,
    X,
    ::ProjectionRetraction,
) where {n}
    return retract!(Sphere(n), q, p, X)
end
function Manifolds.local_metric(
    ::MetricManifold{ℝ,TestSphere{n},<:TestSphericalMetric},
    p,
    B::DefaultOrthonormalBasis{ℝ,<:ManifoldsBase.TangentSpaceType},
) where {n}
    return Diagonal(ones(n)) # TODO: fix?
end
function Manifolds.get_vector!(
    ::MetricManifold{ℝ,<:TestSphere{N},<:TestSphericalMetric},
    Y,
    p,
    c,
    ::DefaultOrthonormalBasis{ℝ,<:ManifoldsBase.TangentSpaceType},
) where {N}
    Y .= 1
    return Y # this is just a dummy to check that dispatch works
end
@testset "Test ODE setup for computing geodesics" begin
    M = TestSphere{2}()
    p = [0.0, 0.0, 1.0]
    X = π / (2 * sqrt(2)) .* [0.0, 1.0, 1.0]
    M2 = MetricManifold(M, TestSphericalMetric())
    @test_throws ErrorException exp(M, p, X)
    @test_throws ErrorException exp(M2, p, X)
    using OrdinaryDiffEq
    exp(M2, p, X)
end
