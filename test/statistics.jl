include("header.jl")
using StatsBase: AbstractWeights, pweights
using Random: GLOBAL_RNG, seed!
import ManifoldsBase:
    active_traits,
    manifold_dimension,
    exp!,
    log!,
    inner,
    zero_vector!,
    decorated_manifold,
    base_manifold,
    get_embedding
using Manifolds:
    AbstractApproximationMethod,
    CyclicProximalPointEstimation,
    GeodesicInterpolation,
    GeodesicInterpolationWithinRadius,
    GradientDescentEstimation,
    WeiszfeldEstimation
import Manifolds:
    mean, mean!, median, median!, var, mean_and_var, default_approximation_method

struct TestStatsSphere{N} <: AbstractManifold{ℝ} end
TestStatsSphere(N) = TestStatsSphere{N}()
manifold_dimension(::TestStatsSphere{N}) where {N} = manifold_dimension(Sphere(N))
function exp!(::TestStatsSphere{N}, q, p, X; kwargs...) where {N}
    return exp!(Sphere(N), q, p, X; kwargs...)
end
function Manifolds.exp_fused!(::TestStatsSphere{N}, q, p, X, t::Number; kwargs...) where {N}
    return Manifolds.exp_fused!(Sphere(N), q, p, X, t; kwargs...)
end
function log!(::TestStatsSphere{N}, X, p, q; kwargs...) where {N}
    return log!(Sphere(N), X, p, q; kwargs...)
end
function inner(::TestStatsSphere{N}, p, X, Y; kwargs...) where {N}
    return inner(Sphere(N), p, X, Y; kwargs...)
end
function zero_vector!(::TestStatsSphere{N}, X, p; kwargs...) where {N}
    return zero_vector!(Sphere(N), X, p; kwargs...)
end

struct TestStatsEuclidean{N} <: AbstractManifold{ℝ} end
TestStatsEuclidean(N) = TestStatsEuclidean{N}()
manifold_dimension(::TestStatsEuclidean{N}) where {N} = manifold_dimension(Euclidean(N))
function exp!(::TestStatsEuclidean{N}, q, p, X; kwargs...) where {N}
    return exp!(Euclidean(N), q, p, X; kwargs...)
end
function Manifolds.exp_fused!(
    ::TestStatsEuclidean{N},
    q,
    p,
    X,
    t::Number;
    kwargs...,
) where {N}
    return Manifolds.exp_fused!(Euclidean(N), q, p, X, t; kwargs...)
end
function log!(::TestStatsEuclidean{N}, X, p, q; kwargs...) where {N}
    return log!(Euclidean(N), X, p, q; kwargs...)
end
function inner(::TestStatsEuclidean{N}, p, X, Y; kwargs...) where {N}
    return inner(Euclidean(N), p, X, Y; kwargs...)
end
function zero_vector!(::TestStatsEuclidean{N}, X, p; kwargs...) where {N}
    return zero_vector!(Euclidean(N), X, p; kwargs...)
end

struct TestStatsNotImplementedEmbeddedManifold <: AbstractDecoratorManifold{ℝ} end
function active_traits(f, ::TestStatsNotImplementedEmbeddedManifold, args...)
    return merge_traits(IsEmbeddedSubmanifold())
end
decorated_manifold(::TestStatsNotImplementedEmbeddedManifold) = Sphere(2)
get_embedding(::TestStatsNotImplementedEmbeddedManifold) = Sphere(2)
base_manifold(::TestStatsNotImplementedEmbeddedManifold) = Sphere(2)

struct TestStatsNotImplementedEmbeddedManifold2 <: AbstractDecoratorManifold{ℝ} end
function active_traits(f, ::TestStatsNotImplementedEmbeddedManifold2, args...)
    return merge_traits(IsIsometricEmbeddedManifold())
end
decorated_manifold(::TestStatsNotImplementedEmbeddedManifold2) = Sphere(2)
get_embedding(::TestStatsNotImplementedEmbeddedManifold2) = Sphere(2)
base_manifold(::TestStatsNotImplementedEmbeddedManifold2) = Sphere(2)

struct TestStatsNotImplementedEmbeddedManifold3 <: AbstractDecoratorManifold{ℝ} end
function active_traits(f, ::TestStatsNotImplementedEmbeddedManifold3, args...)
    return merge_traits(IsEmbeddedManifold())
end
decorated_manifold(::TestStatsNotImplementedEmbeddedManifold3) = Sphere(2)
get_embedding(::TestStatsNotImplementedEmbeddedManifold3) = Sphere(2)
base_manifold(::TestStatsNotImplementedEmbeddedManifold3) = Sphere(2)

function test_mean(M, x, yexp=nothing, method...; kwargs...)
    @testset "mean unweighted" begin
        y = mean(M, x; kwargs...)
        @test is_point(M, y; atol=10^-9)
        if yexp !== nothing
            @test isapprox(M, y, yexp; atol=10^-7)
        end
        y2, _ = mean_and_var(M, x; kwargs...)
        @test isapprox(M, y2, y; atol=10^-7)
        y3, _ = mean_and_std(M, x; kwargs...)
        @test isapprox(M, y3, y; atol=10^-7)
    end

    @testset "mean weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))
        y = mean(M, x; kwargs...)
        for w in (w1, w2, w3)
            @test is_point(M, mean(M, x, w; kwargs...); atol=10^-9)
            @test isapprox(M, mean(M, x, w; kwargs...), y)

            @test isapprox(M, mean_and_var(M, x, w; kwargs...)[1], y; atol=10^-7)
            @test isapprox(M, mean_and_std(M, x, w; kwargs...)[1], y; atol=10^-7)
        end
        @test_throws DimensionMismatch mean(M, x, pweights(ones(n + 1)); kwargs...)
        @test_throws DimensionMismatch mean!(
            M,
            y,
            x,
            pweights(ones(n + 1)),
            Manifolds.default_approximation_method(M, mean);
            kwargs...,
        )
    end
    return nothing
end

function test_median(
    M,
    x,
    yexp=nothing;
    method::Union{Nothing,AbstractApproximationMethod}=nothing,
    kwargs...,
)
    @testset "median unweighted$(!isnothing(method) ? " ($method)" : "")" begin
        y = isnothing(method) ? median(M, x; kwargs...) : median(M, x, method; kwargs...)
        @test is_point(M, y; atol=10^-9)
        if yexp !== nothing
            @test isapprox(M, y, yexp; atol=5 * 10^-5)
        end
    end

    @testset "median weighted$(!isnothing(method) ? " ($method)" : "")" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))
        y = median(M, x; kwargs...)
        for w in (w1, w2, w3)
            if isnothing(method)
                @test is_point(M, median(M, x, w; kwargs...); atol=10^-9)
            else
                @test is_point(M, median(M, x, w, method; kwargs...); atol=10^-9)
            end
            @test isapprox(M, median(M, x, w; kwargs...), y; atol=10^-4)
        end
        if isnothing(method)
            @test_throws Exception median(M, x, pweights(ones(n + 1)); kwargs...)
        else
            @test_throws Exception median(M, x, pweights(ones(n + 1)), method; kwargs...)
        end
    end
    return nothing
end

function test_var(M, x, vexp=nothing; kwargs...)
    n = length(x)
    @testset "var unweighted" begin
        v = var(M, x; kwargs...)
        if vexp !== nothing
            @test v ≈ vexp
        end
        @test v ≈ var(M, x; corrected=true, kwargs...)
        _, v2 = mean_and_var(M, x; kwargs...)
        @test v2 ≈ v
        m = mean(M, x; kwargs...)
        @test var(M, x, m; kwargs...) ≈ var(M, x; kwargs...)
        @test var(M, x; corrected=false, kwargs...) ≈
              var(M, x, m; corrected=false, kwargs...)
        @test var(M, x; corrected=false, kwargs...) ≈ var(M, x; kwargs...) * (n - 1) / n
        @test var(M, x, m; corrected=false, kwargs...) ≈
              var(M, x, m; kwargs...) * (n - 1) / n
    end

    @testset "var weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))

        v = var(M, x; corrected=false, kwargs...)
        for w in (w1, w2, w3)
            @test var(M, x, w; kwargs...) ≈ v
            @test var(M, x, w; corrected=false, kwargs...) ≈ v
            @test mean_and_var(M, x, w; kwargs...)[2] ≈ var(M, x, w; kwargs...)
            m = mean(M, x, w; kwargs...)
            @test var(M, x, w, m; kwargs...) ≈ var(M, x, w; kwargs...)
            @test var(M, x, w; corrected=true, kwargs...) ≈
                  var(M, x, w, m; corrected=true, kwargs...)
            @test var(M, x, w; corrected=true, kwargs...) ≈
                  var(M, x, w; kwargs...) * n / (n - 1)
            @test var(M, x, w, m; corrected=true, kwargs...) ≈
                  var(M, x, w, m; kwargs...) * n / (n - 1)
        end
        @test_throws DimensionMismatch var(M, x, pweights(ones(n + 1)); kwargs...)
        @test_throws DimensionMismatch mean_and_var(
            M,
            x,
            pweights(ones(n + 1)),
            GeodesicInterpolation();
            kwargs...,
        )
    end
    return nothing
end

function test_std(M, x, sexp=nothing; kwargs...)
    n = length(x)
    @testset "std unweighted" begin
        s = std(M, x; kwargs...)
        if sexp !== nothing
            @test s ≈ sexp
        end
        @test s ≈ std(M, x; corrected=true, kwargs...)
        @test s ≈ √var(M, x; corrected=true, kwargs...)
        _, s2 = mean_and_std(M, x; kwargs...)
        @test s2 ≈ s
        m = mean(M, x; kwargs...)
        @test std(M, x, m; kwargs...) ≈ std(M, x; kwargs...)
        @test std(M, x; corrected=false, kwargs...) ≈
              std(M, x, m; corrected=false, kwargs...)
        @test std(M, x; corrected=false, kwargs...) ≈
              std(M, x; kwargs...) * sqrt((n - 1) / n)
        @test std(M, x, m; corrected=false, kwargs...) ≈
              std(M, x, m; kwargs...) * sqrt((n - 1) / n)
    end

    @testset "std weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))

        s = std(M, x; corrected=false, kwargs...)
        for w in (w1, w2, w3)
            @test std(M, x, w; kwargs...) ≈ s
            @test std(M, x, w; corrected=false, kwargs...) ≈ s
            @test mean_and_std(M, x, w; kwargs...)[2] ≈ std(M, x, w; kwargs...)
            m = mean(M, x, w; kwargs...)
            @test std(M, x, w, m; kwargs...) ≈ std(M, x, w; kwargs...)
            @test std(M, x, w; corrected=true, kwargs...) ≈
                  std(M, x, w, m; corrected=true, kwargs...)
            @test std(M, x, w; corrected=true, kwargs...) ≈
                  std(M, x, w; kwargs...) * sqrt(n / (n - 1))
            @test std(M, x, w, m; corrected=true, kwargs...) ≈
                  std(M, x, w, m; kwargs...) * sqrt(n / (n - 1))
        end
        @test_throws DimensionMismatch std(M, x, pweights(ones(n + 1)); kwargs...)
    end
    return nothing
end

function test_moments(M, x)
    n = length(x)
    @testset "moments unweighted" begin
        m = mean(M, x)
        for i in 1:5
            @test moment(M, x, i) ≈ mean(distance.(Ref(M), Ref(m), x) .^ i)
            @test moment(M, x, i, m) ≈ moment(M, x, i)
        end
        @test moment(M, x, 2) ≈ var(M, x; corrected=false)
        @test skewness(M, x) ≈ moment(M, x, 3) / moment(M, x, 2)^(3 / 2)
        @test kurtosis(M, x) ≈ moment(M, x, 4) / moment(M, x, 2)^2 - 3

        @test moment(M, x, 2, m) ≈ var(M, x; corrected=false)
        @test skewness(M, x, m) ≈ moment(M, x, 3) / moment(M, x, 2)^(3 / 2)
        @test kurtosis(M, x, m) ≈ moment(M, x, 4) / moment(M, x, 2)^2 - 3
    end

    @testset "moments weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))

        for w in (w1, w2, w3)
            m = mean(M, x, w)
            for i in 1:5
                @test moment(M, x, i, w) ≈ mean(distance.(Ref(M), Ref(m), x) .^ i, w)
                @test moment(M, x, i, w, m) ≈ moment(M, x, i, w)
            end
            @test moment(M, x, 2, w) ≈ var(M, x, w; corrected=false)
            @test skewness(M, x, w) ≈ moment(M, x, 3, w) / moment(M, x, 2, w)^(3 / 2)
            @test kurtosis(M, x, w) ≈ moment(M, x, 4, w) / moment(M, x, 2, w)^2 - 3

            @test moment(M, x, 2, w, m) ≈ var(M, x, w; corrected=false)
            @test skewness(M, x, w, m) ≈ moment(M, x, 3, w) / moment(M, x, 2, w)^(3 / 2)
            @test kurtosis(M, x, w, m) ≈ moment(M, x, 4, w) / moment(M, x, 2, w)^2 - 3
        end
        @test_throws DimensionMismatch moment(M, x, 3, pweights(ones(n + 1)))
        @test_throws DimensionMismatch skewness(M, x, pweights(ones(n + 1)))
        @test_throws DimensionMismatch kurtosis(M, x, pweights(ones(n + 1)))
    end
    return nothing
end

struct TestStatsOverload1 <: AbstractManifold{ℝ} end
struct TestStatsMethod1 <: AbstractApproximationMethod end

function mean!(
    ::TestStatsOverload1,
    y,
    ::AbstractVector,
    ::AbstractWeights,
    ::GradientDescentEstimation,
)
    return fill!(y, 3)
end
function mean(
    ::TestStatsOverload1,
    ::AbstractVector,
    ::AbstractWeights,
    ::GradientDescentEstimation,
)
    return fill(3, 1)
end

function median(
    ::TestStatsOverload1,
    ::AbstractVector,
    ::AbstractWeights,
    ::CyclicProximalPointEstimation,
)
    return fill(3, 1)
end
function median!(
    ::TestStatsOverload1,
    y,
    ::AbstractVector,
    ::AbstractWeights,
    ::CyclicProximalPointEstimation,
)
    return fill!(y, 3)
end

function var(::TestStatsOverload1, ::AbstractVector, ::AbstractWeights, m; corrected=false)
    return 4 + 5 * corrected
end
function mean_and_var(
    ::TestStatsOverload1,
    ::AbstractVector,
    ::AbstractWeights;
    corrected=false,
    kwargs...,
)
    return [4.0], 4 + 5 * corrected
end
function mean_and_var(
    ::TestStatsOverload1,
    ::AbstractVector,
    ::AbstractWeights,
    ::TestStatsMethod1;
    corrected=false,
    kwargs...,
)
    return [5.0], 9 + 7 * corrected
end

@testset "Statistics" begin
    @testset "defaults and overloads" begin
        w = pweights([2.0])
        x = [[0.0]]
        @testset "mean" begin
            M = TestStatsOverload1()
            y = similar(x[1])
            @test mean(M, x) == [3.0]
            @test mean!(M, y, x) == [3.0]
            @test mean(M, x, w) == [3.0]
            @test mean(M, x, w, GradientDescentEstimation()) == [3.0]
            @test mean!(M, y, x, w, GradientDescentEstimation()) == [3.0]
            @test mean(M, x, GradientDescentEstimation()) == [3.0]
            @test mean!(M, y, x, GradientDescentEstimation()) == [3.0]
        end

        @testset "median" begin
            M = TestStatsOverload1()
            @test median(M, x) == [3.0]
            @test median(M, x, w) == [3.0]
            @test median(M, x, w, CyclicProximalPointEstimation()) == [3.0]
            @test median(M, x, CyclicProximalPointEstimation()) == [3.0]
        end

        @testset "var" begin
            M = TestStatsOverload1()
            @test mean_and_var(M, x) == ([3.0], 9)
            @test mean_and_var(M, x, w) == ([4.0], 4)
            @test mean_and_std(M, x) == ([3.0], 3.0)
            @test mean_and_std(M, x, w) == ([4.0], 2.0)
            @test var(M, x) == 9
            @test var(M, x, 2) == 9
            @test var(M, x, w) == 4
            @test var(M, x, w, 2) == 4
            @test std(M, x) == 3.0
            @test std(M, x, 2) == 3.0
            @test std(M, x, w) == 2.0
            @test std(M, x, w, 2) == 2.0

            @test Manifolds.default_approximation_method(M, mean_and_std) ==
                  Manifolds.default_approximation_method(M, mean)
            @test mean_and_var(M, x, TestStatsMethod1()) == ([5.0], 16)
            @test mean_and_var(M, x, w, TestStatsMethod1()) == ([5.0], 9)
            @test mean_and_std(M, x, TestStatsMethod1()) == ([5.0], 4.0)
            @test mean_and_std(M, x, w, TestStatsMethod1()) == ([5.0], 3.0)
        end
    end

    @testset "decorator dispatch" begin
        # equality tests are intentional to ensure correct dispatch
        # (both calls eventually use the same method)
        ps = [normalize([1, 0, 0] .+ 0.1 .* randn(3)) for _ in 1:3]
        M1 = TestStatsNotImplementedEmbeddedManifold()
        @test mean!(M1, similar(ps[1]), ps) == mean!(Sphere(2), similar(ps[1]), ps)
        @test mean(M1, ps) == mean(Sphere(2), ps)

        M2 = TestStatsNotImplementedEmbeddedManifold2()
        @test_throws MethodError mean(M2, ps)
        @test_throws MethodError mean!(M2, similar(ps[1]), ps)
        @test_throws MethodError median(M2, ps)
        @test_throws MethodError median!(M2, similar(ps[1]), ps)

        M3 = TestStatsNotImplementedEmbeddedManifold3()
        @test_throws MethodError mean(M3, ps)
        @test_throws MethodError mean!(M3, similar(ps[1]), ps)
        @test_throws MethodError median(M3, ps)
        @test_throws MethodError median!(M3, similar(ps[1]), ps)
    end

    @testset "TestStatsSphere" begin
        M = TestStatsSphere(2)

        @testset "consistency" begin
            p = [0.0, 0.0, 1.0]
            n = 3
            x = [
                exp(M, p, π / 6 * [cos(α), sin(α), 0.0]) for
                α in range(0, 2 * π - 2 * π / n, length=n)
            ]
            test_mean(M, x)
            test_median(M, x; atol=1e-12)
            test_median(M, x; method=CyclicProximalPointEstimation(), atol=1e-12)
            test_median(M, x; method=WeiszfeldEstimation())
            test_var(M, x)
            test_std(M, x)
            test_moments(M, x)
        end

        @testset "zero variance" begin
            x = fill([0.0, 0.0, 1.0], 5)
            test_mean(M, x, [0.0, 0.0, 1.0])
            test_median(M, x, [0.0, 0.0, 1.0]; atol=10^-12)
            test_var(M, x, 0.0)
            test_std(M, x, 0.0)
        end

        @testset "three points" begin
            x = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0] / √2, [0.0, 1.0, 0.0]]
            θ = π / 4
            @test isapprox(M, mean(M, x), geodesic(M, x[1], x[3], θ))
            test_mean(M, x, [1.0, 1.0, 0.0] / √2)
            test_median(M, x, [1.0, 1.0, 0.0] / √2; atol=10^-12)
            test_var(M, x, θ^2)
            test_std(M, x, θ)
            test_moments(M, x)
        end
    end

    @testset "TestStatsEuclidean{N}" begin
        @testset "N=1" begin
            M = TestStatsEuclidean(1)

            rng = MersenneTwister(42)
            x = [randn(rng, 1) for _ in 1:10]
            vx = vcat(x...)

            test_mean(M, x, mean(x))
            test_median(M, x; atol=10^-12)
            test_var(M, x, var(vx))
            test_std(M, x, std(vx))
            test_moments(M, x)

            w = pweights(rand(rng, 10))
            @test mean(M, x, w) ≈ [mean(vx, w)]
            @test var(M, x, w) ≈ var(vx, w)
            @test std(M, x, w) ≈ std(vx, w)
        end

        @testset "N=5" begin
            M = TestStatsEuclidean(5)

            rng = MersenneTwister(42)
            x = [randn(rng, 5) for _ in 1:10]
            test_mean(M, x, mean(x))
            test_var(M, x, sum(var(x)))
            test_std(M, x, sqrt(sum(std(x) .^ 2)))
            test_moments(M, x)

            w = pweights(rand(rng, 10))
            ax = hcat(x...)
            @test mean(M, x, w) ≈ mean(ax, w; dims=2)
            @test var(M, x, w) ≈ sum(var(ax, w, 2))
            @test std(M, x, w) ≈ sqrt(sum(std(ax, w, 2) .^ 2))
        end
    end

    @testset "Euclidean statistics" begin
        @testset "N=1" begin
            M = Euclidean()
            @testset "scalar" begin
                x = [1.0, 2.0, 3.0, 4.0]
                w = pweights(ones(length(x)) / length(x))
                @test mean(M, x) ≈ mean(x)
                @test mean(Euclidean(; parameter=:field), x) ≈ mean(x)
                @test mean(M, x, w) ≈ mean(x, w)
                @test median(M, x; rng=MersenneTwister(1212), atol=10^-12) ≈ median(x)
                @test median(M, x, w; rng=MersenneTwister(1212), atol=10^-12) ≈ median(x, w)
                @test var(M, x) ≈ var(x)
                @test var(M, x, w) ≈ var(x, w)
                @test std(M, x) ≈ std(x)
                @test std(M, x, w) ≈ std(x, w)

                test_mean(M, x)
                test_median(M, x; atol=10^-12)
                test_var(M, x)
                test_std(M, x)
                test_moments(M, x)
            end

            @testset "vector" begin
                x = [fill(1.0), fill(2.0), fill(3.0), fill(4.0)]
                vx = vcat(x...)
                w = pweights(ones(length(x)) / length(x))
                @test mean(M, x) ≈ mean(x)
                @test mean(M, x, w) ≈ [mean(vx, w)]
                @test var(M, x) ≈ var(vx)
                @test var(M, x, w) ≈ var(vx, w)
                @test std(M, x) ≈ std(vx)
                @test std(M, x, w) ≈ std(vx, w)

                test_mean(M, x)
                test_median(M, x; atol=10^-12)
                test_var(M, x)
                test_std(M, x)
                test_moments(M, x)
                y = copy(x[1])
                mean!(M, y, x)
                @test y == mean(x)
            end
        end
    end

    @testset "GeodesicInterpolation" begin
        @testset "mean" begin
            rng = MersenneTwister(1212)
            @testset "exact for Euclidean" begin
                M = Euclidean(2, 2)
                x = [randn(rng, 2, 2) for _ in 1:10]
                w = pweights([rand(rng) for _ in 1:10])
                @test mean(M, x, GeodesicInterpolation()) ≈ mean(x)
                @test mean(M, x, w, GeodesicInterpolation()) ≈
                      mean(M, x, w, GradientDescentEstimation())
            end

            @testset "three points" begin
                S = Sphere(2)
                x = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0] / √2]
                @test mean(S, [x[1], x[1]], GeodesicInterpolation()) ≈ x[1]
                @test mean(S, x[1:2], GeodesicInterpolation()) ≈ x[3]
                @test mean(S, x[1:2], pweights([1, 0]), GeodesicInterpolation()) ≈ x[1]
                @test mean(S, x[1:2], pweights([0, 1]), GeodesicInterpolation()) ≈ x[2]
                @test mean(S, x[1:2], pweights([1, 2]), GeodesicInterpolation()) ≈
                      shortest_geodesic(S, x[1], x[2], 2 / 3)
                m = mean(S, x, pweights([1, 2, 3]), GeodesicInterpolation())
                @test m ≈ shortest_geodesic(
                    S,
                    shortest_geodesic(S, x[1], x[2], 2 / 3),
                    x[3],
                    1 / 2,
                )
            end

            @testset "Zero weights" begin
                M = Sphere(2)
                a = sqrt(2)
                pV = [[0.0, 0.0, 1.0], [1 / a, 1 / a, 0], [-1 / a, 1 / a, 0.0]]
                for method in [GeodesicInterpolation(), GradientDescentEstimation()]
                    @testset "with $method" begin
                        m1 = mean(M, pV, [0.0, 0.5, 0.5], method)
                        g1 = shortest_geodesic(M, pV[2], pV[3], 0.5)
                        @test isapprox(M, m1, g1; atol=1e-7)
                        m2 = mean(M, pV, [0.5, 0.0, 0.5], method)
                        @test isapprox(M, m2, shortest_geodesic(M, pV[1], pV[3], 0.5))
                        m3 = mean(M, pV, [0.5, 0.0, 0.0], method)
                        @test isapprox(M, m3, pV[1])
                        m4 = mean(M, pV, [0.0, 1.0, 0.0], method)
                        @test isapprox(M, m4, pV[2])
                        m5 = mean(M, pV, [0.0, 0.0, 1.0], method)
                        @test isapprox(M, m5, pV[3])
                    end
                end
            end

            @testset "resumable" begin
                S = Sphere(2)
                x = [normalize(randn(rng, 3)) for _ in 1:10]
                w = pweights([rand(rng) for _ in 1:10])
                ypart = mean(S, x[1:5], pweights(w[1:5]), GeodesicInterpolation())
                yfull = mean(S, x, w, GeodesicInterpolation())
                x2 = [[ypart]; x[6:end]]
                w2 = pweights([sum(w[1:5]); w[6:end]])
                @test mean(S, x2, w2, GeodesicInterpolation()) ≈ yfull
            end
        end

        @testset "welford" begin
            rng = MersenneTwister(56)

            @testset "exact for Euclidean" begin
                M = TestStatsEuclidean(4)
                x = [randn(rng, 4) for _ in 1:10]
                w = pweights([rand(rng) for _ in 1:10])
                m1, v1 = mean_and_var(M, x, GeodesicInterpolation())
                @test m1 ≈ mean(x)
                @test v1 ≈ sum(var(x))
                m2, v2 = mean_and_var(M, x, w, GeodesicInterpolation())
                @test m2 ≈ mean(M, x, w, GradientDescentEstimation())
                @test v2 ≈ var(M, x, w)
            end

            @testset "three points" begin
                S = Sphere(2)
                x = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0] / √2]
                m1, v1 = mean_and_var(S, [x[1], x[1]], GeodesicInterpolation())
                @test m1 ≈ x[1]
                @test v1 ≈ 0 atol = 1e-6
                m2, v2 = mean_and_var(S, x[1:2], GeodesicInterpolation())
                @test m2 ≈ x[3]
                @test v2 ≈ (π / 4)^2 * 2 / (2 - 1)
                m3, v3 = mean_and_var(S, x[1:2], pweights([1, 0]), GeodesicInterpolation())
                @test m3 ≈ x[1]
                @test v3 ≈ 0 atol = 1e-6
                m4, v4 = mean_and_var(S, x[1:2], pweights([0, 1]), GeodesicInterpolation())
                @test m4 ≈ x[2]
                @test v4 ≈ 0 atol = 1e-6
                m5, v5 = mean_and_var(S, x, pweights([1, 2, 3]), GeodesicInterpolation())
                @test m5 ≈ shortest_geodesic(
                    S,
                    shortest_geodesic(S, x[1], x[2], 2 / 3),
                    x[3],
                    1 / 2,
                )
                @test v5 ≈ var(S, x, pweights([1, 2, 3]), m5)
            end

            @testset "within radius" begin
                rng = MersenneTwister(47)
                @test repr(GeodesicInterpolationWithinRadius(Inf)) ==
                      "GeodesicInterpolationWithinRadius(Inf)"
                @test_throws DomainError GeodesicInterpolationWithinRadius(-0.1)
                S = Sphere(2)
                x = [[1.0, 0, 0], [0, 1.0, 0]]
                m = mean(S, x, GeodesicInterpolation())
                mg = mean(S, x, GradientDescentEstimation(); p0=m)
                vg = var(S, x, mg)

                @test mean(S, x, GeodesicInterpolationWithinRadius(Inf)) == m
                @test mean(S, x, GeodesicInterpolationWithinRadius(π)) == m
                @test mean(S, x, GeodesicInterpolationWithinRadius(π / 8)) != m
                @test mean(S, x, GeodesicInterpolationWithinRadius(π / 8)) == mg

                m, v = mean_and_var(S, x, GeodesicInterpolation())
                @test mean_and_var(S, x, GeodesicInterpolationWithinRadius(Inf)) == (m, v)
                @test mean_and_var(S, x, GeodesicInterpolationWithinRadius(π)) == (m, v)
                @test mean_and_var(S, x, GeodesicInterpolationWithinRadius(π / 8)) != (m, v)
                @test mean_and_var(S, x, GeodesicInterpolationWithinRadius(π / 8)) ==
                      (mg, vg)
            end
        end

        @testset "SymmetricPositiveDefinite default" begin
            rng = MersenneTwister(36)
            P1 = SymmetricPositiveDefinite(3)
            P2 = MetricManifold(P1, AffineInvariantMetric())
            @testset "$P" for P in [P1, P2]
                p0 = collect(exp(Symmetric(randn(rng, 3, 3) * 0.1)))
                x = [exp(P, p0, Symmetric(randn(rng, 3, 3) * 0.1)) for _ in 1:10]
                w = pweights([rand(rng) for _ in 1:length(x)])
                m = mean(P, x, w)
                mg = mean(P, x, w, GeodesicInterpolation())
                mf = mean(P, x, w, GradientDescentEstimation())
                @test m == mg
                @test m != mf
            end
        end

        @testset "Sphere default" begin
            rng = MersenneTwister(47)
            S = Sphere(2)
            p0 = [1.0, 0, 0]
            x = [normalize(randn(rng, 3)) for _ in 1:10]
            x = [x; -x]
            w = pweights([rand(rng) for _ in 1:length(x)])
            m = mean(S, x, w)
            mg = mean(S, x, w, GeodesicInterpolation())
            mf = mean(S, x, w, GradientDescentEstimation(); p0=mg)
            @test m != mg
            @test m == mf

            μ = randn(rng, 3) .* 10
            x = [normalize(randn(rng, 3) .+ μ) for _ in 1:10]
            w = pweights([rand(rng) for _ in 1:length(x)])
            m = mean(S, x, w)
            mg = mean(S, x, w, GeodesicInterpolation())
            @test m == mg
        end

        @testset "ProjectiveSpace default" begin
            rng = MersenneTwister(47)
            M = ProjectiveSpace(2)
            p0 = [1.0, 0, 0]
            x = [normalize(randn(rng, 3)) for _ in 1:10]
            x = [x; -x]
            w = pweights([rand(rng) for _ in 1:length(x)])
            m = mean(M, x, w)
            mg = mean(M, x, w, GeodesicInterpolation())
            mf = mean(M, x, w, GradientDescentEstimation(); p0=mg)
            @test !isapprox(M, m, mg)
            @test isapprox(M, m, mf)

            μ = randn(rng, 3) .* 10
            x = [normalize(randn(rng, 3) .+ μ) for _ in 1:10]
            w = pweights([rand(rng) for _ in 1:length(x)])
            m = mean(M, x, w)
            mg = mean(M, x, w, GeodesicInterpolation())
            @test m == mg
        end

        @testset "Rotations default" begin
            rng = MersenneTwister(47)
            R = Manifolds.Rotations(3)
            p0 = collect(Diagonal(ones(3)))
            X1 = hat(R, p0, [1.0, 0.0, 0.0])
            X2 = hat(R, p0, [0.0, 1.0, 0.0])
            x = [geodesic(R, p0, X1, π / 2 * (1:4)); geodesic(R, p0, X2, π / 2 * (1:4))]
            w = pweights([rand(rng) for _ in 1:length(x)])
            m = mean(R, x, w)
            mg = mean(R, x, w, GeodesicInterpolation())
            mf = mean(R, x, w, GradientDescentEstimation(); p0=mg)
            @test m != mg
            @test m == mf

            μ = project(R, randn(3, 3))
            d = Manifolds.normal_tvector_distribution(R, μ, 0.1)
            x = [exp(R, μ, rand(rng, d)) for _ in 1:10]
            w = pweights([rand(rng) for _ in 1:length(x)])
            m = mean(R, x, w)
            mg = mean(R, x, w, GeodesicInterpolation())
            @test m == mg
        end

        @testset "Grassmann default" begin
            rng = MersenneTwister(47)
            G = Manifolds.Grassmann(3, 2)
            p0 = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            x = [exp(G, p0, project(G, p0, randn(rng, 3, 2) * 10)) for _ in 1:10]
            w = pweights([rand(rng) for _ in 1:length(x)])
            m = mean(G, x, w)
            mg = mean(G, x, w, GeodesicInterpolation())
            mf = mean(G, x, w, GradientDescentEstimation(); p0=mg)
            @test m != mg
            @test m == mf

            x = [exp(G, p0, project(G, p0, randn(rng, 3, 2) * 0.01)) for _ in 1:10]
            w = pweights([rand(rng) for _ in 1:length(x)])
            m = mean(G, x, w)
            mg = mean(G, x, w, GeodesicInterpolation())
            @test m == mg
        end
    end

    @testset "Extrinsic mean" begin
        rng = MersenneTwister(47)
        S = Sphere(2)
        x = [normalize(randn(rng, 3)) for _ in 1:10]
        w = pweights([rand(rng) for _ in 1:length(x)])
        m = normalize(mean(reduce(hcat, x), w; dims=2)[:, 1])
        mg = mean(S, x, w, ExtrinsicEstimation(EfficientEstimator()))
        @test isapprox(S, m, mg)
    end

    @testset "Extrinsic median" begin
        rng = MersenneTwister(47)
        S = Sphere(2)
        x = [normalize(randn(rng, 3)) for _ in 1:10]
        w = pweights([rand(rng) for _ in 1:length(x)])
        m = normalize(median(Euclidean(3), x, w))
        mg = median(S, x, w, ExtrinsicEstimation(CyclicProximalPointEstimation()))
        @test isapprox(S, m, mg)
    end

    @testset "Covariance Default" begin
        @test default_approximation_method(TestStatsSphere{2}(), cov) ==
              GradientDescentEstimation()
    end

    @testset "Covariance matrix, Euclidean" begin
        rng = MersenneTwister(47)
        M = Euclidean(3)
        x = [randn(rng, 3) for _ in 1:10]
        @test isapprox(cov(M, x), cov(x))
        covest = SimpleCovariance(; corrected=false)
        @test isapprox(cov(M, x; tangent_space_covariance_estimator=covest), cov(covest, x))
    end

    @testset "Covariance matrix, sphere" begin
        rng = MersenneTwister(47)
        S = Sphere(2)
        x = [normalize(randn(rng, 3)) for _ in 1:10]
        covm = cov(S, x)
        @test size(covm) == (2, 2)
        @test isposdef(covm)
        @test issymmetric(covm)
    end
end
