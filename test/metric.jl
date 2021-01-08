using FiniteDifferences, ForwardDiff
using LinearAlgebra: I
using StatsBase: AbstractWeights, pweights
import Manifolds: mean!, median!

include("utils.jl")

struct TestEuclidean{N} <: Manifold{ℝ} end
struct TestEuclideanMetric <: Metric end

Manifolds.manifold_dimension(::TestEuclidean{N}) where {N} = N
function Manifolds.local_metric(
    M::MetricManifold{ℝ,<:TestEuclidean,<:TestEuclideanMetric},
    x,
)
    return Diagonal(1.0:manifold_dimension(M))
end

struct TestSphere{N,T} <: Manifold{ℝ}
    r::T
end

struct TestSphericalMetric <: Metric end

Manifolds.manifold_dimension(::TestSphere{N}) where {N} = N
function Manifolds.local_metric(M::MetricManifold{ℝ,<:TestSphere,<:TestSphericalMetric}, x)
    r = base_manifold(M).r
    d = allocate(x)
    d[1] = r^2
    d[2] = d[1] * sin(x[1])^2
    return Diagonal(d)
end
sph_to_cart(θ, ϕ) = [cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)]

struct BaseManifold{N} <: Manifold{ℝ} end
struct BaseManifoldMetric{M} <: Metric end
struct DefaultBaseManifoldMetric <: Metric end
struct NotImplementedMetric <: Metric end

Manifolds.manifold_dimension(::BaseManifold{N}) where {N} = N
Manifolds.inner(::BaseManifold, x, v, w) = 2 * dot(v, w)
Manifolds.exp!(::BaseManifold, y, x, v) = y .= x + 2 * v
Manifolds.log!(::BaseManifold, v, x, y) = v .= (y - x) / 2
Manifolds.project!(::BaseManifold, w, x, v) = w .= 2 .* v
Manifolds.project!(::BaseManifold, y, x) = (y .= x)
Manifolds.injectivity_radius(::BaseManifold) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::Any) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::AbstractRetractionMethod) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::ExponentialRetraction) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::Any, ::AbstractRetractionMethod) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::Any, ::ExponentialRetraction) = Inf
function Manifolds.local_metric(
    ::MetricManifold{ℝ,BaseManifold{N},BaseManifoldMetric{N}},
    x,
) where {N}
    return 2 * one(x * x')
end
function Manifolds.exp!(
    M::MetricManifold{ℝ,BaseManifold{N},BaseManifoldMetric{N}},
    y,
    x,
    v,
) where {N}
    return exp!(base_manifold(M), y, x, v)
end
function Manifolds.vector_transport_to!(::BaseManifold, vto, x, v, y, ::ParallelTransport)
    return (vto .= v)
end
function Manifolds.get_basis(::BaseManifold{N}, x, B::DefaultOrthonormalBasis) where {N}
    return CachedBasis(B, [(Matrix{eltype(x)}(I, N, N)[:, i]) for i in 1:N])
end
Manifolds.get_coordinates!(::BaseManifold, Y, p, X, ::DefaultOrthonormalBasis) = (Y .= X)
Manifolds.get_vector!(::BaseManifold, Y, p, X, ::DefaultOrthonormalBasis) = (Y .= X)
Manifolds.default_metric_dispatch(::BaseManifold, ::DefaultBaseManifoldMetric) = Val(true)
function Manifolds.projected_distribution(M::BaseManifold, d)
    return ProjectedPointDistribution(M, d, project!, rand(d))
end
function Manifolds.projected_distribution(M::BaseManifold, d, x)
    return ProjectedPointDistribution(M, d, project!, x)
end
function Manifolds.mean!(::BaseManifold, y, x::AbstractVector, w::AbstractVector; kwargs...)
    return fill!(y, 1)
end
function Manifolds.median!(
    ::BaseManifold,
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return fill!(y, 2)
end
function Manifolds.mean!(
    ::MetricManifold{ℝ,BaseManifold{N},BaseManifoldMetric{N}},
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {N}
    return fill!(y, 3)
end
function Manifolds.median!(
    ::MetricManifold{ℝ,BaseManifold{N},BaseManifoldMetric{N}},
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {N}
    return fill!(y, 4)
end

function Manifolds.flat!(
    ::BaseManifold,
    v::FVector{Manifolds.CotangentSpaceType},
    x,
    w::FVector{Manifolds.TangentSpaceType},
)
    v.data .= 2 .* w.data
    return v
end
function Manifolds.sharp!(
    ::BaseManifold,
    v::FVector{Manifolds.TangentSpaceType},
    x,
    w::FVector{Manifolds.CotangentSpaceType},
)
    v.data .= w.data ./ 2
    return v
end

@testset "Metrics" begin
    # some tests failed due to insufficient accuracy for a particularly bad RNG state
    Random.seed!(42)
    @testset "Metric Basics" begin
        #one for MetricManifold, one for Manifold & Metric
        @test length(methods(is_default_metric)) == 2
    end

    @testset "solve_exp_ode error message" begin
        E = TestEuclidean{3}()
        g = TestEuclideanMetric()
        M = MetricManifold(E, g)

        x = [1.0, 2.0, 3.0]
        v = [2.0, 3.0, 4.0]
        @test_throws ErrorException exp(M, x, v)
        if VERSION ≥ v"1.1"
            using OrdinaryDiffEq
            exp(M, x, v)
        end
    end
    @testset "Local Metric Error message" begin
        M = MetricManifold(BaseManifold{2}(), NotImplementedMetric())
        @test_throws ErrorException local_metric(M, [3, 4])
    end
    @testset "scaled Euclidean metric" begin
        n = 3
        E = TestEuclidean{n}()
        g = TestEuclideanMetric()
        M = MetricManifold(E, g)
        @test repr(M) == "MetricManifold(TestEuclidean{3}(), TestEuclideanMetric())"

        if VERSION ≥ v"1.3"
            @test TestEuclideanMetric()(E) === M
            @test TestEuclideanMetric(E) === M
        end

        G = Diagonal(1.0:n)
        invG = inv(G)
        @test manifold_dimension(M) == n
        @test base_manifold(M) === E
        @test metric(M) === g

        @test_throws ErrorException local_metric_jacobian(E, zeros(3))
        @test_throws ErrorException christoffel_symbols_second_jacobian(E, zeros(3))

        for vtype in (Vector, MVector{n})
            x, v, w = vtype(randn(n)), vtype(randn(n)), vtype(randn(n))

            @test check_manifold_point(M, x) == check_manifold_point(E, x)
            @test check_tangent_vector(M, x, v) == check_tangent_vector(E, x, v)

            @test local_metric(M, x) ≈ G
            @test inverse_local_metric(M, x) ≈ invG
            @test det_local_metric(M, x) ≈ *(1.0:n...)
            @test log_local_metric_density(M, x) ≈ sum(log.(1.0:n)) / 2
            @test inner(M, x, v, w) ≈ dot(v, G * w) atol = 1e-6
            @test norm(M, x, v) ≈ sqrt(dot(v, G * v)) atol = 1e-6

            if VERSION ≥ v"1.1"
                T = 0:0.5:10
                @test geodesic(M, x, v, T) ≈ [x + t * v for t in T] atol = 1e-6
            end

            @test christoffel_symbols_first(M, x) ≈ zeros(n, n, n) atol = 1e-6
            @test christoffel_symbols_second(M, x) ≈ zeros(n, n, n) atol = 1e-6
            @test riemann_tensor(M, x) ≈ zeros(n, n, n, n) atol = 1e-6
            @test ricci_tensor(M, x) ≈ zeros(n, n) atol = 1e-6
            @test ricci_curvature(M, x) ≈ 0 atol = 1e-6
            @test gaussian_curvature(M, x) ≈ 0 atol = 1e-6
            @test einstein_tensor(M, x) ≈ zeros(n, n) atol = 1e-6

            fdm = FiniteDifferencesBackend(forward_fdm(2, 1))
            @test christoffel_symbols_first(M, x; backend=fdm) ≈ zeros(n, n, n) atol = 1e-6
            @test christoffel_symbols_second(M, x; backend=fdm) ≈ zeros(n, n, n) atol = 1e-6
            @test riemann_tensor(M, x; backend=fdm) ≈ zeros(n, n, n, n) atol = 1e-6
            @test ricci_tensor(M, x; backend=fdm) ≈ zeros(n, n) atol = 1e-6
            @test ricci_curvature(M, x; backend=fdm) ≈ 0 atol = 1e-6
            @test gaussian_curvature(M, x; backend=fdm) ≈ 0 atol = 1e-6
            @test einstein_tensor(M, x; backend=fdm) ≈ zeros(n, n) atol = 1e-6

            fwd_diff = Manifolds.ForwardDiffBackend()
            @test christoffel_symbols_first(M, x; backend=fwd_diff) ≈ zeros(n, n, n) atol =
                1e-6
            @test christoffel_symbols_second(M, x; backend=fwd_diff) ≈ zeros(n, n, n) atol =
                1e-6
            @test riemann_tensor(M, x; backend=fwd_diff) ≈ zeros(n, n, n, n) atol = 1e-6
            @test ricci_tensor(M, x; backend=fwd_diff) ≈ zeros(n, n) atol = 1e-6
            @test ricci_curvature(M, x; backend=fwd_diff) ≈ 0 atol = 1e-6
            @test gaussian_curvature(M, x; backend=fwd_diff) ≈ 0 atol = 1e-6
            @test einstein_tensor(M, x; backend=fwd_diff) ≈ zeros(n, n) atol = 1e-6
        end
    end

    @testset "scaled Sphere metric" begin
        n = 2
        r = 10 * rand()
        θ, ϕ = π * rand(), 2π * rand()
        Sr = TestSphere{n,Float64}(r)
        S = Manifolds.Sphere(n)
        g = TestSphericalMetric()
        M = MetricManifold(Sr, g)

        @test manifold_dimension(M) == n
        @test base_manifold(M) === Sr
        @test metric(M) === g

        for vtype in (Vector, MVector{n})
            x = vtype([θ, ϕ])
            G = Diagonal(vtype([1, sin(θ)^2])) .* r^2
            invG = Diagonal(vtype([1, 1 / sin(θ)^2])) ./ r^2
            v, w = normalize(randn(n)), normalize(randn(n))

            @test local_metric(M, x) ≈ G atol = 1e-6
            @test inverse_local_metric(M, x) ≈ invG atol = 1e-6
            @test det_local_metric(M, x) ≈ r^4 * sin(θ)^2 atol = 1e-6
            @test log_local_metric_density(M, x) ≈ 2 * log(r) + log(sin(θ)) atol = 1e-6
            @test inner(M, x, v, w) ≈ dot(v, G * w) atol = 1e-6
            @test norm(M, x, v) ≈ sqrt(dot(v, G * v)) atol = 1e-6

            xcart = sph_to_cart(θ, ϕ)
            vcart = [
                cos(ϕ)*cos(θ) -sin(ϕ)*sin(θ)
                sin(ϕ)*cos(θ) cos(ϕ)*sin(θ)
                -sin(θ) 0
            ] * v

            if VERSION ≥ v"1.1" && (!Sys.iswindows() || Sys.ARCH == :x86_64)
                @testset "numerically integrated geodesics for $vtype" begin
                    T = 0:0.1:1
                    @test isapprox(
                        [sph_to_cart(yi...) for yi in geodesic(M, x, v, T)],
                        geodesic(S, xcart, vcart, T);
                        atol=1e-3,
                        rtol=1e-3,
                    )
                end
            end

            Γ₁ = christoffel_symbols_first(M, x)
            for i in 1:n, j in 1:n, k in 1:n
                if (i, j, k) == (1, 2, 2) || (i, j, k) == (2, 1, 2)
                    @test Γ₁[i, j, k] ≈ r^2 * cos(θ) * sin(θ) atol = 1e-6
                elseif (i, j, k) == (2, 2, 1)
                    @test Γ₁[i, j, k] ≈ -r^2 * cos(θ) * sin(θ) atol = 1e-6
                else
                    @test Γ₁[i, j, k] ≈ 0 atol = 1e-6
                end
            end

            Γ₂ = christoffel_symbols_second(M, x)
            for l in 1:n, i in 1:n, j in 1:n
                if (l, i, j) == (1, 2, 2)
                    @test Γ₂[l, i, j] ≈ -cos(θ) * sin(θ) atol = 1e-6
                elseif (l, i, j) == (2, 1, 2) || (l, i, j) == (2, 2, 1)
                    @test Γ₂[l, i, j] ≈ cot(θ) atol = 1e-6
                else
                    @test Γ₂[l, i, j] ≈ 0 atol = 1e-6
                end
            end

            R = riemann_tensor(M, x)
            for l in 1:n, i in 1:n, j in 1:n, k in 1:n
                if (l, i, j, k) == (2, 1, 1, 2)
                    @test R[l, i, j, k] ≈ -1 atol = 2e-6
                elseif (l, i, j, k) == (2, 1, 2, 1)
                    @test R[l, i, j, k] ≈ 1 atol = 2e-6
                elseif (l, i, j, k) == (1, 2, 1, 2)
                    @test R[l, i, j, k] ≈ sin(θ)^2 atol = 1e-6
                elseif (l, i, j, k) == (1, 2, 2, 1)
                    @test R[l, i, j, k] ≈ -sin(θ)^2 atol = 1e-6
                else
                    @test R[l, i, j, k] ≈ 0 atol = 1e-6
                end
            end

            @test ricci_tensor(M, x) ≈ G ./ r^2 atol = 2e-6
            @test ricci_curvature(M, x) ≈ 2 / r^2 atol = 2e-6
            @test gaussian_curvature(M, x) ≈ 1 / r^2 atol = 2e-6
            @test einstein_tensor(M, x) ≈ ricci_tensor(M, x) - gaussian_curvature(M, x) .* G atol =
                1e-6
        end
    end

    @testset "Metric decorator" begin
        M = BaseManifold{3}()
        g = BaseManifoldMetric{3}()
        MM = MetricManifold(M, g)
        if VERSION ≥ v"1.3"
            @test DefaultBaseManifoldMetric(BaseManifold{3}()) ===
                  MetricManifold(BaseManifold{3}(), DefaultBaseManifoldMetric())
            MT = DefaultBaseManifoldMetric()
            @test MT(BaseManifold{3}()) ===
                  MetricManifold(BaseManifold{3}(), DefaultBaseManifoldMetric())
        end
        g2 = DefaultBaseManifoldMetric()
        MM2 = MetricManifold(M, g2)

        @test (@inferred Manifolds.default_metric_dispatch(MM)) ===
              (@inferred Manifolds.default_metric_dispatch(base_manifold(MM), metric(MM)))
        @test (@inferred Manifolds.default_metric_dispatch(MM2)) ===
              (@inferred Manifolds.default_metric_dispatch(base_manifold(MM2), metric(MM2)))
        @test (@inferred Manifolds.default_metric_dispatch(MM2)) === Val(true)
        @test is_default_metric(MM) == is_default_metric(base_manifold(MM), metric(MM))
        @test is_default_metric(MM2) == is_default_metric(base_manifold(MM2), metric(MM2))
        @test is_default_metric(MM2)
        @test Manifolds.default_decorator_dispatch(MM) ===
              Manifolds.default_metric_dispatch(MM)
        @test Manifolds.default_decorator_dispatch(MM2) ===
              Manifolds.default_metric_dispatch(MM2)

        @test convert(typeof(MM2), M) == MM2
        @test_throws ErrorException convert(typeof(MM), M)
        x = [0.1, 0.2, 0.4]
        v = [0.5, 0.7, 0.11]
        w = [0.13, 0.17, 0.19]
        y = allocate(x)

        @test inner(M, x, v, w) == 2 * dot(v, w)
        @test inner(MM, x, v, w) === inner(M, x, v, w)
        @test norm(MM, x, v) === norm(M, x, v)
        @test exp(M, x, v) == x + 2 * v
        @test exp(MM2, x, v) == exp(M, x, v)
        @test exp!(MM, y, x, v) === exp!(M, y, x, v)
        @test retract!(MM, y, x, v) === retract!(M, y, x, v)
        @test retract!(MM, y, x, v, 1) === retract!(M, y, x, v, 1)
        # without a definition for the metric from the embedding, no projection possible
        @test_throws ErrorException log!(MM, w, x, y) === project!(M, w, x, y)
        @test_throws ErrorException project!(MM, w, x, v) === project!(M, w, x, v)
        @test_throws ErrorException project!(MM, y, x) === project!(M, y, x)
        @test_throws ErrorException vector_transport_to!(MM, w, x, v, y) ===
                                    vector_transport_to!(M, w, x, v, y)
        # without DiffEq, these error
        # @test_throws ErrorException exp(MM,x, v, 1:3)
        # @test_throws ErrorException exp!(MM, y, x, v)
        # these always fall back anyways.
        @test zero_tangent_vector!(MM, v, x) === zero_tangent_vector!(M, v, x)

        @test injectivity_radius(MM, x) === injectivity_radius(M, x)
        @test injectivity_radius(MM) === injectivity_radius(M)
        @test injectivity_radius(MM, ProjectionRetraction()) ===
              injectivity_radius(M, ProjectionRetraction())
        @test injectivity_radius(MM, ExponentialRetraction()) ===
              injectivity_radius(M, ExponentialRetraction())
        @test injectivity_radius(MM) === injectivity_radius(M)

        @test is_manifold_point(MM, x) === is_manifold_point(M, x)
        @test is_tangent_vector(MM, x, v) === is_tangent_vector(M, x, v)

        @test_throws ErrorException local_metric(MM2, x)
        @test_throws ErrorException local_metric_jacobian(MM2, x)
        @test_throws ErrorException christoffel_symbols_second_jacobian(MM2, x)
        # MM falls back to nondefault error
        @test_throws MethodError projected_distribution(MM, 1, x)
        @test_throws MethodError projected_distribution(MM, 1)

        @test inner(MM2, x, v, w) === inner(M, x, v, w)
        @test norm(MM2, x, v) === norm(M, x, v)
        @test distance(MM2, x, y) === distance(M, x, y)
        @test exp!(MM2, y, x, v) === exp!(M, y, x, v)
        @test exp(MM2, x, v) == exp(M, x, v)
        @test log!(MM2, v, x, y) === log!(M, v, x, y)
        @test log(MM2, x, y) == log(M, x, y)
        @test retract!(MM2, y, x, v) === retract!(M, y, x, v)
        @test retract!(MM2, y, x, v, 1) === retract!(M, y, x, v, 1)

        @test project!(MM2, y, x) === project!(M, y, x)
        @test project!(MM2, w, x, v) === project!(M, w, x, v)
        @test vector_transport_to!(MM2, w, x, v, y) == vector_transport_to!(M, w, x, v, y)
        @test zero_tangent_vector!(MM2, v, x) === zero_tangent_vector!(M, v, x)
        @test injectivity_radius(MM2, x) === injectivity_radius(M, x)
        @test injectivity_radius(MM2) === injectivity_radius(M)
        @test injectivity_radius(MM2, x, ExponentialRetraction()) ===
              injectivity_radius(M, x, ExponentialRetraction())
        @test injectivity_radius(MM2, ExponentialRetraction()) ===
              injectivity_radius(M, ExponentialRetraction())
        @test injectivity_radius(MM2, x, ProjectionRetraction()) ===
              injectivity_radius(M, x, ProjectionRetraction())
        @test injectivity_radius(MM2, ProjectionRetraction()) ===
              injectivity_radius(M, ProjectionRetraction())
        @test is_manifold_point(MM2, x) === is_manifold_point(M, x)
        @test is_tangent_vector(MM2, x, v) === is_tangent_vector(M, x, v)

        a = Manifolds.projected_distribution(M, Distributions.MvNormal(zero(zeros(3)), 1.0))
        b = Manifolds.projected_distribution(
            MM2,
            Distributions.MvNormal(zero(zeros(3)), 1.0),
        )
        @test isapprox(Matrix(a.distribution.Σ), Matrix(b.distribution.Σ))
        @test isapprox(a.distribution.μ, b.distribution.μ)
        @test get_basis(M, x, DefaultOrthonormalBasis()).data ==
              get_basis(MM2, x, DefaultOrthonormalBasis()).data
        @test_throws ErrorException get_basis(MM, x, DefaultOrthonormalBasis())
        cov = flat(M, x, FVector(TangentSpace, v))
        cow = flat(M, x, FVector(TangentSpace, w))
        @test cov.data ≈ flat(MM, x, FVector(TangentSpace, v)).data
        cotspace = CotangentBundleFibers(M)
        cotspace2 = CotangentBundleFibers(MM)
        @test cov.data ≈ 2 * v
        @test inner(M, x, v, w) ≈ inner(cotspace, x, cov.data, cow.data)
        @test inner(MM, x, v, w) ≈ inner(cotspace, x, cov.data, cow.data)
        @test inner(MM, x, v, w) ≈ inner(cotspace2, x, cov.data, cow.data)
        @test sharp(M, x, cov).data ≈ v

        xsample = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        w = pweights([0.5, 0.5])
        # test despatch with results from above
        @test mean(M, xsample, w) ≈ ones(3)
        @test mean(MM2, xsample, w) ≈ ones(3)
        @test mean(MM, xsample, w) ≈ 3 .* ones(3)

        @test median(M, xsample, w) ≈ 2 .* ones(3)
        @test median(MM2, xsample, w) ≈ 2 * ones(3)
        @test median(MM, xsample, w) ≈ 4 .* ones(3)
    end

    @testset "Metric decorator dispatches" begin
        M = BaseManifold{3}()
        g = BaseManifoldMetric{3}()
        MM = MetricManifold(M, g)
        x = [1, 2, 3]
        # nonmutating always go to parent for allocation
        for f in [exp, flat, inverse_retract, log, mean, median, project]
            @test Manifolds.decorator_transparent_dispatch(f, MM) === Val{:parent}()
        end
        for f in [sharp, retract, get_vector, get_coordinates]
            @test Manifolds.decorator_transparent_dispatch(f, MM) === Val{:parent}()
        end
        for f in [vector_transport_along, vector_transport_direction, vector_transport_to]
            @test Manifolds.decorator_transparent_dispatch(f, MM) === Val{:parent}()
        end
        for f in [get_basis, inner]
            @test Manifolds.decorator_transparent_dispatch(f, MM) === Val{:intransparent}()
        end
        for f in [get_coordinates!, get_vector!]
            @test Manifolds.decorator_transparent_dispatch(f, MM) === Val{:intransparent}()
        end

        # mirroring ones are mostly intransparent despite for a few cases - e.g. dispatch/default last variables
        for f in [exp!, flat!, inverse_retract!, log!, mean!, median!]
            @test Manifolds.decorator_transparent_dispatch(f, MM) === Val{:intransparent}()
        end
        for f in [norm, project!, sharp!, retract!]
            @test Manifolds.decorator_transparent_dispatch(f, MM) === Val{:intransparent}()
        end
        for f in [vector_transport_along!, vector_transport_to!]
            @test Manifolds.decorator_transparent_dispatch(f, MM) === Val{:intransparent}()
        end
        @test Manifolds.decorator_transparent_dispatch(vector_transport_direction!, MM) ===
              Val{:parent}()

        @test Manifolds.decorator_transparent_dispatch(exp!, MM, x, x, x, x) ===
              Val{:parent}()
        @test Manifolds.decorator_transparent_dispatch(
            inverse_retract!,
            MM,
            x,
            x,
            x,
            LogarithmicInverseRetraction(),
        ) === Val{:parent}()
        @test Manifolds.decorator_transparent_dispatch(
            retract!,
            MM,
            x,
            x,
            x,
            ExponentialRetraction(),
        ) === Val{:parent}()
    end
end
