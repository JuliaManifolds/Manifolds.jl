include("../utils.jl")
include("group_utils.jl")

using OrdinaryDiffEq
import Manifolds: invariant_metric_dispatch, default_metric_dispatch, local_metric

struct TestInvariantMetricBase <: AbstractMetric end

function local_metric(
    ::MetricManifold{𝔽,<:AbstractManifold,TestInvariantMetricBase},
    ::Identity,
    ::DefaultOrthonormalBasis,
) where {𝔽}
    return Diagonal([1.0, 2.0, 3.0])
end

struct TestBiInvariantMetricBase <: AbstractMetric end

function local_metric(
    ::MetricManifold{𝔽,<:AbstractManifold,<:TestBiInvariantMetricBase},
    ::Identity,
    ::DefaultOrthonormalBasis,
) where {𝔽}
    return Diagonal(0.4I, 3)
end

struct TestInvariantMetricManifold <: AbstractManifold{ℝ} end

struct TestDefaultInvariantMetricManifold <: AbstractManifold{ℝ} end

function ManifoldsBase.active_traits(f, ::TestDefaultInvariantMetricManifold, args...)
    return merge_traits(HasRightInvariantMetric())
end

@testset "Invariant metrics" begin
    base_metric = TestInvariantMetricBase()

    e = Matrix{Float64}(I, 3, 3)
    @testset "inner/norm" begin
        SO3 = SpecialOrthogonal(3)
        p = exp(hat(SO3, Identity(SO3), [1.0, 2.0, 3.0]))

        B = DefaultOrthonormalBasis()

        fX = ManifoldsBase.TFVector([2.0, 3.0, 4.0], B)
        fY = ManifoldsBase.TFVector([3.0, 4.0, 1.0], B)
        X = hat(SO3, Identity(SO3), fX.data)
        Y = hat(SO3, Identity(SO3), fY.data)

        G = MetricManifold(SO3, base_metric)
        @test inner(G, p, fX, fY) ≈ dot(fX.data, Diagonal([1.0, 2.0, 3.0]) * fY.data)
        @test norm(G, p, fX) ≈ sqrt(inner(G, p, fX, fX))
    end

    @testset "log/exp bi-invariant" begin
        SO3 = SpecialOrthogonal(3)
        e = Identity(SO3)
        pe = identity_element(SO3)
        p = exp(hat(SO3, pe, [1.0, 2.0, 3.0]))
        q = exp(hat(SO3, pe, [3.0, 4.0, 1.0]))
        X = hat(SO3, e, [2.0, 3.0, 4.0])

        G = MetricManifold(SO3, TestBiInvariantMetricBase())
        @test isapprox(SO3, exp(G, p, X), exp(SO3, p, X))
        @test isapprox(SO3, p, log(G, p, q), log(SO3, p, q); atol=1e-6)

        G = MetricManifold(SO3, TestBiInvariantMetricBase())
        @test isapprox(SO3, exp(G, p, X), exp(SO3, p, X))
        @test isapprox(SO3, p, log(G, p, q), log(SO3, p, q); atol=1e-6)
    end
end
