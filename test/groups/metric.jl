include("../utils.jl")
include("group_utils.jl")

using OrdinaryDiffEq
import Manifolds: invariant_metric_dispatch, default_metric_dispatch, local_metric

struct TestInvariantMetricBase <: Metric end

function local_metric(
    ::MetricManifold{𝔽,<:Manifold,TestInvariantMetricBase},
    ::Identity,
) where {𝔽}
    return Diagonal([1.0, 2.0, 3.0])
end
function local_metric(
    ::MetricManifold{𝔽,<:Manifold,<:InvariantMetric{TestInvariantMetricBase}},
    p,
) where {𝔽}
    return Diagonal([1.0, 2.0, 3.0])
end

struct TestBiInvariantMetricBase <: Metric end

function invariant_metric_dispatch(
    ::MetricManifold{𝔽,<:Manifold,<:InvariantMetric{TestBiInvariantMetricBase}},
    ::ActionDirection,
) where {𝔽}
    return Val(true)
end

function local_metric(
    ::MetricManifold{𝔽,<:Manifold,<:TestBiInvariantMetricBase},
    ::Identity,
) where {𝔽}
    return Diagonal(0.4I, 3)
end

struct TestInvariantMetricManifold <: Manifold{ℝ} end

struct TestDefaultInvariantMetricManifold <: Manifold{ℝ} end

function default_metric_dispatch(
    ::MetricManifold{
        ℝ,
        TestDefaultInvariantMetricManifold,
        RightInvariantMetric{TestInvariantMetricBase},
    },
)
    return Val(true)
end

invariant_metric_dispatch(::TestDefaultInvariantMetricManifold, ::RightAction) = Val(true)

@testset "Invariant metrics" begin
    base_metric = TestInvariantMetricBase()
    metric = InvariantMetric(base_metric)
    lmetric = LeftInvariantMetric(base_metric)
    rmetric = RightInvariantMetric(base_metric)

    @test InvariantMetric(base_metric) === InvariantMetric(base_metric, LeftAction())
    @test lmetric === InvariantMetric(base_metric, LeftAction())
    @test rmetric === InvariantMetric(base_metric, RightAction())
    @test sprint(show, lmetric) == "LeftInvariantMetric(TestInvariantMetricBase())"
    @test sprint(show, rmetric) == "RightInvariantMetric(TestInvariantMetricBase())"

    @test direction(lmetric) === LeftAction()
    @test direction(rmetric) === RightAction()

    G = MetricManifold(TestInvariantMetricManifold(), lmetric)
    @test (@inferred invariant_metric_dispatch(G, LeftAction())) === Val(true)
    @test (@inferred invariant_metric_dispatch(G, RightAction())) === Val(false)

    G = MetricManifold(TestInvariantMetricManifold(), rmetric)
    @test (@inferred invariant_metric_dispatch(G, RightAction())) === Val(true)
    @test (@inferred invariant_metric_dispatch(G, LeftAction())) === Val(false)

    @test Manifolds.invariant_metric_dispatch(
        TestInvariantMetricManifold(),
        RightAction(),
    ) === Val{false}()
    @test Manifolds.invariant_metric_dispatch(
        TestInvariantMetricManifold(),
        LeftAction(),
    ) === Val{false}()

    G = MetricManifold(
        TestDefaultInvariantMetricManifold(),
        LeftInvariantMetric(TestInvariantMetricBase()),
    )
    @test !is_default_metric(G)
    G = MetricManifold(
        TestDefaultInvariantMetricManifold(),
        RightInvariantMetric(TestInvariantMetricBase()),
    )
    @test is_default_metric(G)

    e = Matrix{Float64}(I, 3, 3)
    @testset "inner/norm" begin
        SO3 = SpecialOrthogonal(3)
        p = exp(hat(SO3, Identity(SO3, e), [1.0, 2.0, 3.0]))
        X = hat(SO3, Identity(SO3, e), [2.0, 3.0, 4.0])
        Y = hat(SO3, Identity(SO3, e), [3.0, 4.0, 1.0])

        G = MetricManifold(SO3, lmetric)
        @test inner(G, p, X, Y) ≈ dot(X, Diagonal([1.0, 2.0, 3.0]) * Y)
        @test norm(G, p, X) ≈ sqrt(inner(G, p, X, X))

        G = MetricManifold(SO3, rmetric)
        @test inner(G, p, X, Y) ≈ dot(p * X * p', Diagonal([1.0, 2.0, 3.0]) * p * Y * p')
        @test norm(G, p, X) ≈ sqrt(inner(G, p, X, X))
    end

    @testset "log/exp bi-invariant" begin
        SO3 = SpecialOrthogonal(3)
        p = exp(hat(SO3, Identity(SO3, e), [1.0, 2.0, 3.0]))
        q = exp(hat(SO3, Identity(SO3, e), [3.0, 4.0, 1.0]))
        X = hat(SO3, Identity(SO3, e), [2.0, 3.0, 4.0])

        G = MetricManifold(SO3, InvariantMetric(TestBiInvariantMetricBase(), LeftAction()))
        @test isapprox(SO3, exp(G, p, X), exp(SO3, p, X))
        @test isapprox(SO3, p, log(G, p, q), log(SO3, p, q); atol = 1e-6)

        G = MetricManifold(SO3, InvariantMetric(TestBiInvariantMetricBase(), RightAction()))
        @test isapprox(SO3, exp(G, p, X), exp(SO3, p, X))
        @test isapprox(SO3, p, log(G, p, q), log(SO3, p, q); atol = 1e-6)
    end

    @testset "exp τ-invariant" begin
        T3 = TranslationGroup(3)
        p = [1.0, 2.0, 3.0]
        X = [3.0, 5.0, 6.0]
        @test isapprox(T3, exp(MetricManifold(T3, lmetric), p, X), p .+ X)
        @test isapprox(T3, exp(MetricManifold(T3, rmetric), p, X), p .+ X)
    end
end
