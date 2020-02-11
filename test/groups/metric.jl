using OrdinaryDiffEq
import Manifolds: has_invariant_metric, is_default_metric, local_metric

struct TestInvariantMetricBase <: Metric end

function local_metric(::MetricManifold{<:Manifold,TestInvariantMetricBase}, ::Identity)
    return Diagonal([1.0, 2.0, 3.0])
end
function local_metric(::MetricManifold{<:Manifold,<:InvariantMetric{TestInvariantMetricBase}}, p)
    return Diagonal([1.0, 2.0, 3.0])
end

struct TestBiInvariantMetricBase <: Metric end

function has_invariant_metric(
    ::MetricManifold{<:Manifold,<:InvariantMetric{TestBiInvariantMetricBase}},
    ::ActionDirection,
)
    return Val(true)
end

function local_metric(::MetricManifold{<:Manifold,<:TestBiInvariantMetricBase}, ::Identity)
    return Diagonal(0.4I, 3)
end

struct TestInvariantMetricManifold <: Manifold end

struct TestDefaultInvariantMetricManifold <: Manifold end

function is_default_metric(
    ::MetricManifold{TestDefaultInvariantMetricManifold,TestInvariantMetricBase},
)
    return true
end

has_invariant_metric(::TestDefaultInvariantMetricManifold, ::RightAction) = Val(true)

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
    @test has_invariant_metric(G, LeftAction()) === Val(true)
    @test has_invariant_metric(G, RightAction()) === Val(false)

    G = MetricManifold(TestInvariantMetricManifold(), rmetric)
    @test has_invariant_metric(G, RightAction()) === Val(true)
    @test has_invariant_metric(G, LeftAction()) === Val(false)

    G = MetricManifold(TestDefaultInvariantMetricManifold(),LeftInvariantMetric(TestInvariantMetricBase()))
    @test !is_default_metric(G)
    G = MetricManifold(TestDefaultInvariantMetricManifold(),RightInvariantMetric(TestInvariantMetricBase()))
    @test is_default_metric(G)

    @testset "inner/norm" begin
        SO3 = SpecialOrthogonal(3)
        p = exp(hat(SO3, Identity(SO3), [1.0, 2.0, 3.0]))
        X = hat(SO3, Identity(SO3), [2.0, 3.0, 4.0])
        Y = hat(SO3, Identity(SO3), [3.0, 4.0, 1.0])

        G = MetricManifold(SO3, lmetric)
        @test inner(G, p, X, Y) ≈ dot(X, Diagonal([1.0, 2.0, 3.0]) * Y)
        @test norm(G, p, X) ≈ sqrt(inner(G, p, X, X))

        G = MetricManifold(SO3, rmetric)
        @test inner(G, p, X, Y) ≈ dot(p * X * p', Diagonal([1.0, 2.0, 3.0]) * p * Y * p')
        @test norm(G, p, X) ≈ sqrt(inner(G, p, X, X))
    end

    @testset "log/exp bi-invariant" begin
        SO3 = SpecialOrthogonal(3)
        p = exp(hat(SO3, Identity(SO3), [1.0, 2.0, 3.0]))
        q = exp(hat(SO3, Identity(SO3), [3.0, 4.0, 1.0]))
        X = hat(SO3, Identity(SO3), [2.0, 3.0, 4.0])

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
