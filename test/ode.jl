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
retract!(::TestSphere{n}, q, p, X) where {n} = retract!(Sphere(n), q, p, X)
function local_metric(
    ::TestSphere{n},
    p,
    B::DefaultOrthonormalBasis{ℝ,<:ManifoldsBase.TangentSpaceType},
) where {n}
    return local_metric(Sphere(n), p, B)
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
