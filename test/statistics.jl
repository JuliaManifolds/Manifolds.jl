include("utils.jl")
using StatsBase: AbstractWeights, pweights
using Random: GLOBAL_RNG, seed!
import ManifoldsBase: manifold_dimension, exp!, log!, distance, zero_tangent_vector!
using Manifolds: AbstractMethod, GradientMethod, CyclicProximalPointMethod
import Manifolds: mean!, median!, var, mean_and_var

struct TestStatsSphere{N} <: Manifold end
TestStatsSphere(N) = TestStatsSphere{N}()
manifold_dimension(M::TestStatsSphere{N}) where {N} = manifold_dimension(Sphere(N))
exp!(M::TestStatsSphere{N}, w, x, y; kwargs...) where {N} = exp!(Sphere(N), w, x, y; kwargs...)
log!(M::TestStatsSphere{N}, y, x, v; kwargs...) where {N} = log!(Sphere(N), y, x, v; kwargs...)
distance(M::TestStatsSphere{N}, x, y; kwargs...) where {N} = distance(Sphere(N), x, y; kwargs...)
zero_tangent_vector!(M::TestStatsSphere{N},  v, x; kwargs...) where {N} = zero_tangent_vector!(Sphere(N),  v, x; kwargs...)

struct TestStatsEuclidean{N} <: Manifold end
TestStatsEuclidean(N) = TestStatsEuclidean{N}()
manifold_dimension(M::TestStatsEuclidean{N}) where {N} = manifold_dimension(Euclidean(N))
exp!(M::TestStatsEuclidean{N}, y, x, v; kwargs...) where {N} = exp!(Euclidean(N), y, x, v; kwargs...)
log!(M::TestStatsEuclidean{N}, w, x, y; kwargs...) where {N} = log!(Euclidean(N), w, x, y; kwargs...)
distance(M::TestStatsEuclidean{N}, x, y; kwargs...) where {N} = distance(Euclidean(N), x, y; kwargs...)
zero_tangent_vector!(M::TestStatsEuclidean{N},  v, x; kwargs...) where {N} = zero_tangent_vector!(Euclidean(N),  v, x; kwargs...)

function test_mean(M, x, yexp = nothing; kwargs...)
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
            end
        end
    end
end
