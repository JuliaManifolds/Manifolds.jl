include("utils.jl")

using FiniteDifferences, ForwardDiff
using LinearAlgebra: I
import Manifolds: retract!
import ManifoldsBase: manifold_dimension, default_retraction_method

#
# Part I: Euclidean
#
struct TestODEEuclidean{N} <: AbstractManifold{ℝ} end
struct TestODEEuclideanMetric <: AbstractMetric end
#
# Part II Spherical
#
struct TestODESphere{N} <: AbstractManifold{ℝ} end
struct TestODESphericalMetric <: AbstractMetric end

function default_retraction_method(
    ::MetricManifold{ℝ,<:TestODESphere,<:TestODESphericalMetric},
)
    return ProjectionRetraction()
end

manifold_dimension(::TestODESphere{N}) where {N} = N
function Manifolds.retract!(
    ::MetricManifold{ℝ,<:TestODESphere{N},<:TestODESphericalMetric},
    q,
    p,
    X,
    ::ProjectionRetraction,
) where {N}
    return retract!(Sphere(N), q, p, X)
end
function Manifolds.local_metric(
    ::MetricManifold{ℝ,<:TestODESphere{N},<:TestODESphericalMetric},
    p,
    B::DefaultOrthonormalBasis{ℝ,<:ManifoldsBase.TangentSpaceType},
) where {N}
    return Manifolds.local_metric(MetricManifold(Sphere(N), EuclideanMetric()), p, B)
end
function Manifolds.get_coordinates!(
    ::MetricManifold{ℝ,<:TestODESphere{N},<:TestODESphericalMetric},
    c,
    p,
    X,
    B::DefaultOrthonormalBasis{ℝ,<:ManifoldsBase.TangentSpaceType},
) where {N}
    return get_coordinates!(Sphere(N), c, p, X, B)
end
function Manifolds.get_vector!(
    ::MetricManifold{ℝ,<:TestODESphere{N},<:TestODESphericalMetric},
    Y,
    p,
    c,
    B::DefaultOrthonormalBasis{ℝ,<:ManifoldsBase.TangentSpaceType},
) where {N}
    return get_vector!(Sphere(N), Y, p, c, B)
end
#@testset "Test ODE setup for computing geodesics" begin
M = TestODESphere{2}()
p = [0.0, 0.0, 1.0]
X = π / (2 * sqrt(2)) .* [0.0, 1.0, 1.0]
M2 = MetricManifold(M, TestODESphericalMetric())
#    @test_throws ErrorException exp(M, p, X)
#    @test_throws ErrorException exp(M2, p, X)
using OrdinaryDiffEq
exp(M2, p, X)
#end
