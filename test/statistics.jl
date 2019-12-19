include("utils.jl")
using StatsBase: AbstractWeights, pweights
using Random: GLOBAL_RNG, seed!
import ManifoldsBase: manifold_dimension, exp!, log!, inner, zero_tangent_vector!
using Manifolds: AbstractMethod, GradientMethod, CyclicProximalPointMethod, GeodesicInterpolationMethod
import Manifolds: mean!, median!, var, mean_and_var

struct TestStatsSphere{N} <: Manifold end
TestStatsSphere(N) = TestStatsSphere{N}()
manifold_dimension(M::TestStatsSphere{N}) where {N} = manifold_dimension(Sphere(N))
exp!(M::TestStatsSphere{N}, w, x, y; kwargs...) where {N} = exp!(Sphere(N), w, x, y; kwargs...)
log!(M::TestStatsSphere{N}, y, x, v; kwargs...) where {N} = log!(Sphere(N), y, x, v; kwargs...)
inner(M::TestStatsSphere{N}, x, v, w; kwargs...) where {N} = inner(Sphere(N), x, v, w; kwargs...)
zero_tangent_vector!(M::TestStatsSphere{N},  v, x; kwargs...) where {N} = zero_tangent_vector!(Sphere(N),  v, x; kwargs...)

struct TestStatsEuclidean{N} <: Manifold end
TestStatsEuclidean(N) = TestStatsEuclidean{N}()
manifold_dimension(M::TestStatsEuclidean{N}) where {N} = manifold_dimension(Euclidean(N))
exp!(M::TestStatsEuclidean{N}, y, x, v; kwargs...) where {N} = exp!(Euclidean(N), y, x, v; kwargs...)
log!(M::TestStatsEuclidean{N}, w, x, y; kwargs...) where {N} = log!(Euclidean(N), w, x, y; kwargs...)
inner(M::TestStatsEuclidean{N}, x, v, w; kwargs...) where {N} = inner(Euclidean(N), x, v, w; kwargs...)
zero_tangent_vector!(M::TestStatsEuclidean{N},  v, x; kwargs...) where {N} = zero_tangent_vector!(Euclidean(N),  v, x; kwargs...)

function test_mean(M, x, yexp = nothing, method...; kwargs...)
    @testset "mean unweighted" begin
        y = mean(M, x; kwargs...)
        @test is_manifold_point(M, y; atol=10^-9)
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
            @test is_manifold_point(M, mean(M, x, w; kwargs...); atol=10^-9)
            @test isapprox(M, mean(M, x, w; kwargs...), y)

            @test isapprox(M, mean_and_var(M, x, w; kwargs...)[1], y; atol=10^-7)
            @test isapprox(M, mean_and_std(M, x, w; kwargs...)[1], y; atol=10^-7)
        end
        @test_throws DimensionMismatch mean(M, x, pweights(ones(n + 1)); kwargs...)
    end
end

function test_median(M, x, yexp = nothing; kwargs...)
    @testset "median unweighted" begin
        y = median(M, x; kwargs...)
        @test is_manifold_point(M, y; atol=10^-9)
        if yexp !== nothing
            @test isapprox(M, y, yexp; atol=10^-5)
        end
    end

    @testset "median weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))
        y = median(M, x; kwargs...)
        for w in (w1, w2, w3)
            @test is_manifold_point(M, median(M, x, w; kwargs...); atol = 10^-9)
            @test isapprox(M, median(M, x, w; kwargs...), y; atol = 10^-4)
        end
        @test_throws Exception median(M, x, pweights(ones(n + 1)); kwargs...)
    end
end

function test_var(M, x, vexp = nothing; kwargs...)
    n = length(x)
    @testset "var unweighted" begin
        v = var(M, x; kwargs...)
        if vexp !== nothing
            @test v ≈ vexp
        end
        @test v ≈ var(M, x; corrected = true, kwargs...)
        _, v2 = mean_and_var(M, x; kwargs...)
        @test v2 ≈ v
        m = mean(M, x; kwargs...)
        @test var(M, x, m; kwargs...) ≈ var(M, x; kwargs...)
        @test var(M, x; corrected = false, kwargs...) ≈ var(M, x, m; corrected = false, kwargs...)
        @test var(M, x; corrected = false, kwargs...) ≈ var(M, x; kwargs...) * (n - 1) / n
        @test var(M, x, m; corrected = false, kwargs...) ≈ var(M, x, m; kwargs...) * (n - 1) / n
    end

    @testset "var weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))

        v = var(M, x; corrected = false, kwargs...)
        for w in (w1, w2, w3)
            @test var(M, x, w; kwargs...) ≈ v
            @test var(M, x, w; corrected = false, kwargs...) ≈ v
            @test mean_and_var(M, x, w; kwargs...)[2] ≈ var(M, x, w; kwargs...)
            m = mean(M, x, w; kwargs...)
            @test var(M, x, w, m; kwargs...) ≈ var(M, x, w; kwargs...)
            @test var(M, x, w; corrected = true, kwargs...) ≈ var(M, x, w, m; corrected = true, kwargs...)
            @test var(M, x, w; corrected = true, kwargs...) ≈ var(M, x, w; kwargs...) * n / (n - 1)
            @test var(M, x, w, m; corrected = true, kwargs...) ≈ var(M, x, w, m; kwargs...) * n / (n - 1)
        end
        @test_throws DimensionMismatch var(M, x, pweights(ones(n + 1)); kwargs...)
    end
end

function test_std(M, x, sexp = nothing; kwargs...)
    n = length(x)
    @testset "std unweighted" begin
        s = std(M, x; kwargs...)
        if sexp !== nothing
            @test s ≈ sexp
        end
        @test s ≈ std(M, x; corrected = true, kwargs...)
        @test s ≈ √var(M, x; corrected = true, kwargs...)
        _, s2 = mean_and_std(M, x; kwargs...)
        @test s2 ≈ s
        m = mean(M, x; kwargs...)
        @test std(M, x, m; kwargs...) ≈ std(M, x; kwargs...)
        @test std(M, x; corrected = false, kwargs...) ≈ std(M, x, m; corrected = false, kwargs...)
        @test std(M, x; corrected = false, kwargs...) ≈ std(M, x; kwargs...) * sqrt((n - 1) / n)
        @test std(M, x, m; corrected = false, kwargs...) ≈ std(M, x, m; kwargs...) * sqrt((n - 1) / n)
    end

    @testset "std weighted" begin
        n = length(x)
        w1 = pweights(ones(n) / n)
        w2 = pweights(ones(n))
        w3 = pweights(2 * ones(n))

        s = std(M, x; corrected = false, kwargs...)
        for w in (w1, w2, w3)
            @test std(M, x, w; kwargs...) ≈ s
            @test std(M, x, w; corrected = false, kwargs...) ≈ s
            @test mean_and_std(M, x, w; kwargs...)[2] ≈ std(M, x, w; kwargs...)
            m = mean(M, x, w; kwargs...)
            @test std(M, x, w, m; kwargs...) ≈ std(M, x, w; kwargs...)
            @test std(M, x, w; corrected = true, kwargs...) ≈ std(M, x, w, m; corrected = true, kwargs...)
            @test std(M, x, w; corrected = true, kwargs...) ≈ std(M, x, w; kwargs...) * sqrt(n / (n - 1))
            @test std(M, x, w, m; corrected = true, kwargs...) ≈ std(M, x, w, m; kwargs...) * sqrt(n / (n - 1))
        end
        @test_throws DimensionMismatch std(M, x, pweights(ones(n + 1)); kwargs...)
    end
end

function test_moments(M, x)
    n = length(x)
    @testset "moments unweighted" begin
        m = mean(M, x)
        for i in 1:5
            @test moment(M, x, i) ≈ mean(distance.(Ref(M), Ref(m), x).^i)
            @test moment(M, x, i, m) ≈ moment(M, x, i)
        end
        @test moment(M, x, 2) ≈ var(M, x; corrected = false)
        @test skewness(M, x) ≈ moment(M, x, 3) / moment(M, x, 2)^(3/2)
        @test kurtosis(M, x) ≈ moment(M, x, 4) / moment(M, x, 2)^2 - 3

        @test moment(M, x, 2, m) ≈ var(M, x; corrected = false)
        @test skewness(M, x, m) ≈ moment(M, x, 3) / moment(M, x, 2)^(3/2)
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
                @test moment(M, x, i, w) ≈ mean(distance.(Ref(M), Ref(m), x).^i, w)
                @test moment(M, x, i, w, m) ≈ moment(M, x, i, w)
            end
            @test moment(M, x, 2, w) ≈ var(M, x, w; corrected = false)
            @test skewness(M, x, w) ≈ moment(M, x, 3, w) / moment(M, x, 2, w)^(3/2)
            @test kurtosis(M, x, w) ≈ moment(M, x, 4, w) / moment(M, x, 2, w)^2 - 3

            @test moment(M, x, 2, w, m) ≈ var(M, x, w; corrected = false)
            @test skewness(M, x, w, m) ≈ moment(M, x, 3, w) / moment(M, x, 2, w)^(3/2)
            @test kurtosis(M, x, w, m) ≈ moment(M, x, 4, w) / moment(M, x, 2, w)^2 - 3
        end
        @test_throws DimensionMismatch moment(M, x, 3, pweights(ones(n + 1)))
        @test_throws DimensionMismatch skewness(M, x, pweights(ones(n + 1)))
        @test_throws DimensionMismatch kurtosis(M, x, pweights(ones(n + 1)))
    end
end

struct TestStatsOverload1 <: Manifold end
struct TestStatsOverload2 <: Manifold end
struct TestStatsOverload3 <: Manifold end
struct TestStatsMethod1 <: AbstractMethod end

mean!(M::TestStatsOverload1, y, x::AbstractVector, w::AbstractWeights, method::GradientMethod) = fill!(y, 3)
mean!(M::TestStatsOverload2, y, x::AbstractVector, w::AbstractWeights) = fill!(y, 4)
mean!(M::TestStatsOverload2, y, x::AbstractVector, w::AbstractWeights, method::GradientMethod) = fill!(y, 3)
mean!(M::TestStatsOverload3, y, x::AbstractVector, w::AbstractWeights, method::TestStatsMethod1 = TestStatsMethod1()) = fill!(y, 5)

median!(M::TestStatsOverload1, y, x::AbstractVector, w::AbstractWeights, method::CyclicProximalPointMethod) = fill!(y, 3)
median!(M::TestStatsOverload2, y, x::AbstractVector, w::AbstractWeights) = fill!(y, 4)
median!(M::TestStatsOverload2, y, x::AbstractVector, w::AbstractWeights, method::CyclicProximalPointMethod) = fill!(y, 3)
median!(M::TestStatsOverload3, y, x::AbstractVector, w::AbstractWeights, method::TestStatsMethod1 = TestStatsMethod1()) = fill!(y, 5)

var(M::TestStatsOverload1, x::AbstractVector, w::AbstractWeights, m; corrected = false) = 4 + 5*corrected
mean_and_var(M::TestStatsOverload1, x::AbstractVector, w::AbstractWeights; corrected = false, kwargs...) = [4.0], 4 + 5*corrected
mean_and_var(M::TestStatsOverload1, x::AbstractVector, w::AbstractWeights, ::TestStatsMethod1; corrected = false, kwargs...) = [5.0], 9 + 7*corrected

@testset "Statistics" begin
    @testset "defaults and overloads" begin
        w = pweights([2.0])
        x = [[0.0]]
        @testset "mean" begin
            M = TestStatsOverload1()
            @test mean(M, x) == [3.0]
            @test mean(M, x, w) == [3.0]
            @test mean(M, x, w, GradientMethod()) == [3.0]
            @test mean(M, x, GradientMethod()) == [3.0]
            M = TestStatsOverload2()
            @test mean(M, x) == [4.0]
            @test mean(M, x, w) == [4.0]
            @test mean(M, x, w, GradientMethod()) == [3.0]
            @test mean(M, x, GradientMethod()) == [3.0]
            M = TestStatsOverload3()
            @test mean(M, x) == [5.0]
            @test mean(M, x, w) == [5.0]
        end

        @testset "median" begin
            M = TestStatsOverload1()
            @test median(M, x) == [3.0]
            @test median(M, x, w) == [3.0]
            @test median(M, x, w, CyclicProximalPointMethod()) == [3.0]
            @test median(M, x, CyclicProximalPointMethod()) == [3.0]
            M = TestStatsOverload2()
            @test median(M, x) == [4.0]
            @test median(M, x, w) == [4.0]
            @test median(M, x, w, CyclicProximalPointMethod()) == [3.0]
            @test median(M, x, CyclicProximalPointMethod()) == [3.0]
            M = TestStatsOverload3()
            @test median(M, x) == [5.0]
            @test median(M, x, w) == [5.0]
        end

        @testset "var" begin
            M = TestStatsOverload1()
            @test mean_and_var(M, x) == ([4.0], 9)
            @test mean_and_var(M, x, w) == ([4.0], 4)
            @test mean_and_std(M, x) == ([4.0], 3.0)
            @test mean_and_std(M, x, w) == ([4.0], 2.0)
            @test var(M, x) == 9
            @test var(M, x, 2) == 9
            @test var(M, x, w) == 4
            @test var(M, x, w, 2) == 4
            @test std(M, x) == 3.0
            @test std(M, x, 2) == 3.0
            @test std(M, x, w) == 2.0
            @test std(M, x, w, 2) == 2.0

            @test mean_and_var(M, x, TestStatsMethod1()) == ([5.0], 16)
            @test mean_and_var(M, x, w, TestStatsMethod1()) == ([5.0], 9)
            @test mean_and_std(M, x, TestStatsMethod1()) == ([5.0], 4.0)
            @test mean_and_std(M, x, w, TestStatsMethod1()) == ([5.0], 3.0)
        end
    end

    @testset "TestStatsSphere" begin
        M = TestStatsSphere(2)

        @testset "consistency" begin
            p = [0.,0.,1.]
            n=3
            x = [ exp(M,p,π/6*[cos(α), sin(α), 0.]) for α = range(0,2*π - 2*π/n, length=n) ]
            test_mean(M, x)
            test_median(M, x; atol = 10^-12)
            test_var(M, x)
            test_std(M, x)
            test_moments(M, x)
        end

        @testset "zero variance" begin
            x = fill([0.,0.,1.], 5)
            test_mean(M, x, [0.,0.,1.])
            test_median(M, x, [0.,0.,1.]; atol = 10^-12)
            test_var(M, x, 0.0)
            test_std(M, x, 0.0)
        end

        @testset "three points" begin
            x = [ [1., 0., 0.], [1.0, 1.0, 0.0] / √2, [0., 1., 0.] ]
            θ = π / 4
            @test isapprox(M, mean(M, x), geodesic(M, x[1], x[3], θ))
            test_mean(M, x, [1.0, 1.0, 0.0] / √2)
            test_median(M, x, [1.0, 1.0, 0.0] / √2; atol = 10^-12)
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
            test_median(M, x; atol = 10^-12)
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
            test_std(M, x, sqrt(sum(std(x).^2)))
            test_moments(M, x)

            w = pweights(rand(rng, 10))
            ax = hcat(x...)
            @test mean(M, x, w) ≈ mean(ax, w; dims=2)
            @test var(M, x, w) ≈ sum(var(ax, w, 2))
            @test std(M, x, w) ≈ sqrt(sum(std(ax, w, 2).^2))
        end
    end

    @testset "Euclidean statistics" begin
        @testset "N=1" begin
            M = Euclidean(1)
            @testset "scalar" begin
                x = [1., 2., 3., 4.,]
                w = pweights(  ones(length(x)) / length(x)  )
                @test mean(M,x) ≈ mean(x)
                @test mean(M,x,w) ≈ mean(x,w)
                @test median(M,x; rng = MersenneTwister(1212), atol = 10^-12) ≈ median(x)
                @test median(M,x,w; rng = MersenneTwister(1212), atol = 10^-12) ≈ median(x,w)
                @test var(M,x) ≈ var(x)
                @test var(M,x,w) ≈ var(x,w)
                @test std(M,x) ≈ std(x)
                @test std(M,x,w) ≈ std(x,w)

                test_mean(M, x)
                test_median(M, x; atol = 10^-12)
                test_var(M, x)
                test_std(M, x)
                test_moments(M, x)
            end

            @testset "vector" begin
                x = [[1.], [2.], [3.], [4.]]
                vx = vcat(x...)
                w = pweights(  ones(length(x)) / length(x)  )
                @test mean(M,x) ≈ mean(x)
                @test mean(M,x,w) ≈ [mean(vx,w)]
                @test var(M,x) ≈ var(vx)
                @test var(M,x,w) ≈ var(vx,w)
                @test std(M,x) ≈ std(vx)
                @test std(M,x,w) ≈ std(vx,w)

                test_mean(M, x)
                test_median(M, x; atol = 10^-12)
                test_var(M, x)
                test_std(M, x)
                test_moments(M, x)
            end
        end
    end

    @testset "GeodesicInterpolationMethod" begin
        @testset "mean" begin
            rng = MersenneTwister(1212)
            @testset "exact for Euclidean" begin
                M = Euclidean(2, 2)
                x = [randn(rng, 2, 2) for _ = 1:10]
                w = pweights([rand(rng) for _ = 1:10])
                @test mean(M, x, GeodesicInterpolationMethod()) ≈ mean(x)
                @test mean(M, x, w, GeodesicInterpolationMethod()) ≈ mean(M, x, w, GradientMethod())
            end

            @testset "three points" begin
                S = Sphere(2)
                x = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0] / √2]
                @test mean(S, [x[1], x[1]], GeodesicInterpolationMethod()) ≈ x[1]
                @test mean(S, x[1:2], GeodesicInterpolationMethod()) ≈ x[3]
                @test mean(S, x[1:2], pweights([1, 0]), GeodesicInterpolationMethod()) ≈ x[1]
                @test mean(S, x[1:2], pweights([0, 1]), GeodesicInterpolationMethod()) ≈ x[2]
                @test mean(S, x[1:2], pweights([1, 2]), GeodesicInterpolationMethod()) ≈ shortest_geodesic(S, x[1], x[2], 2/3)
                m = mean(S, x, pweights([1, 2, 3]), GeodesicInterpolationMethod())
                @test m ≈ shortest_geodesic(S, shortest_geodesic(S, x[1], x[2], 2/3), x[3], 1/2)
            end

            @testset "resumable" begin
                S = Sphere(2)
                x = [normalize(randn(rng, 3)) for _= 1:10]
                w = pweights([rand(rng) for _ = 1:10])
                ypart = mean(S, x[1:5], pweights(w[1:5]), GeodesicInterpolationMethod())
                yfull = mean(S, x, w, GeodesicInterpolationMethod())
                x2 = [[ypart]; x[6:end]]
                w2 = pweights([sum(w[1:5]); w[6:end]])
                @test mean(S, x2, w2, GeodesicInterpolationMethod()) ≈ yfull
            end
        end

        @testset "welford" begin
            rng = MersenneTwister(56)

            @testset "exact for Euclidean" begin
                M = TestStatsEuclidean(4)
                x = [randn(rng, 4) for _ = 1:10]
                w = pweights([rand(rng) for _ = 1:10])
                m1, v1 = mean_and_var(M, x, GeodesicInterpolationMethod())
                @test m1 ≈ mean(x)
                @test v1 ≈ sum(var(x))
                m2, v2 = mean_and_var(M, x, w, GeodesicInterpolationMethod())
                @test m2 ≈ mean(M, x, w, GradientMethod())
                @test v2 ≈ var(M, x, w)
            end

            @testset "three points" begin
                S = Sphere(2)
                x = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0] / √2]
                m1, v1 = mean_and_var(S, [x[1], x[1]], GeodesicInterpolationMethod())
                @test m1 ≈ x[1]
                @test v1 ≈ 0 atol=1e-6
                m2, v2 = mean_and_var(S, x[1:2], GeodesicInterpolationMethod())
                @test m2 ≈ x[3]
                @test v2 ≈ (π/4)^2 * 2 / (2 - 1)
                m3, v3 = mean_and_var(S, x[1:2], pweights([1, 0]), GeodesicInterpolationMethod())
                @test m3 ≈ x[1]
                @test v3 ≈ 0 atol=1e-6
                m4, v4 = mean_and_var(S, x[1:2], pweights([0, 1]), GeodesicInterpolationMethod())
                @test m4 ≈ x[2]
                @test v4 ≈ 0 atol=1e-6
                m5, v5 = mean_and_var(S, x, pweights([1, 2, 3]), GeodesicInterpolationMethod())
                @test m5 ≈ shortest_geodesic(S, shortest_geodesic(S, x[1], x[2], 2/3), x[3], 1/2)
                @test v5 ≈ var(S, x, pweights([1, 2, 3]), m5)
            end
        end

        @testset "SymmetricPositiveDefinite default" begin
            rng = MersenneTwister(36)
            P = SymmetricPositiveDefinite(3)
            x0 = collect(exp(Symmetric(randn(rng, 3, 3) * 0.1)))
            x = [exp(P, x0, Symmetric(randn(rng, 3, 3) * 0.1)) for _=1:10]
            w = pweights([rand(rng) for _ = 1:length(x)])
            m = mean(P, x, w)
            mg = mean(P, x, w, GeodesicInterpolationMethod())
            mf = mean(P, x, w, GradientMethod())
            @test m == mg
            @test m != mf
        end

        @testset "Sphere default" begin
            rng = MersenneTwister(47)
            S = Sphere(2)
            x0 = [1.0, 0, 0]
            x = [normalize(randn(rng, 3)) for _=1:10]
            x = [x; -x]
            w = pweights([rand(rng) for _ = 1:length(x)])
            m = mean(S, x, w)
            mg = mean(S, x, w, GeodesicInterpolationMethod())
            mf = mean(S, x, w, GradientMethod(); x0 = mg)
            @test m != mg
            @test m == mf

            μ = randn(rng, 3) .* 10
            x = [normalize(randn(rng, 3) .+ μ) for _=1:10]
            w = pweights([rand(rng) for _ = 1:length(x)])
            m = mean(S, x, w)
            mg = mean(S, x, w, GeodesicInterpolationMethod())
            @test m == mg
        end

        @testset "Rotations default" begin
            rng = MersenneTwister(47)
            R = Manifolds.Rotations(3)
            x0 = collect(Diagonal(ones(3)))
            v1 = hat(R, x0, [1.0, 0.0, 0.0])
            v2 = hat(R, x0, [0.0, 1.0, 0.0])
            x = [exp(R, x0, v1, π/2*(1:4)); exp(R, x0, v2, π/2*(1:4))]
            w = pweights([rand(rng) for _ = 1:length(x)])
            m = mean(R, x, w)
            mg = mean(R, x, w, GeodesicInterpolationMethod())
            mf = mean(R, x, w, GradientMethod(); x0 = mg)
            @test m != mg
            @test m == mf

            μ = project_point(R, randn(3, 3))
            d = normal_tvector_distribution(R, μ, 0.1)
            x = [exp(R, μ, rand(rng, d)) for _ = 1:10]
            w = pweights([rand(rng) for _ = 1:length(x)])
            m = mean(R, x, w)
            mg = mean(R, x, w, GeodesicInterpolationMethod())
            @test m == mg
        end

        @testset "Grassmann default" begin
            rng = MersenneTwister(47)
            G = Manifolds.Grassmann(3, 2)
            x0 = [1.0 0.0; 0.0 1.0; 0.0 0.0]
            x = [exp(G, x0, project_tangent(G, x0, randn(rng, 3, 2) * 10)) for _= 1:10]
            w = pweights([rand(rng) for _ = 1:length(x)])
            m = mean(G, x, w)
            mg = mean(G, x, w, GeodesicInterpolationMethod())
            mf = mean(G, x, w, GradientMethod(); x0 = mg)
            @test m != mg
            @test m == mf

            x = [exp(G, x0, project_tangent(G, x0, randn(rng, 3, 2) * 0.01)) for _= 1:10]
            w = pweights([rand(rng) for _ = 1:length(x)])
            m = mean(G, x, w)
            mg = mean(G, x, w, GeodesicInterpolationMethod())
            @test m == mg
        end
    end
end
