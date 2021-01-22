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
    A::Manifolds.AbstractAtlas,
    i,
    p,
)
    return Diagonal(1.0:manifold_dimension(M))
end

struct TestSphere{N,T} <: Manifold{ℝ}
    r::T
end

struct TestSphericalMetric <: Metric end

Manifolds.manifold_dimension(::TestSphere{N}) where {N} = N
function Manifolds.local_metric(
    M::MetricManifold{ℝ,<:TestSphere,<:TestSphericalMetric},
    A::Manifolds.AbstractAtlas,
    i,
    p,
)
    r = base_manifold(M).r
    d = allocate(p)
    d[1] = r^2
    d[2] = d[1] * sin(p[1])^2
    return Diagonal(d)
end
sph_to_cart(θ, ϕ) = [cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)]

struct BaseManifold{N} <: Manifold{ℝ} end
struct BaseManifoldMetric{M} <: Metric end
struct DefaultBaseManifoldMetric <: Metric end
struct NotImplementedMetric <: Metric end

Manifolds.manifold_dimension(::BaseManifold{N}) where {N} = N
Manifolds.inner(::BaseManifold, p, X, Y) = 2 * dot(X, Y)
Manifolds.exp!(::BaseManifold, q, p, X) = q .= p + 2 * X
Manifolds.log!(::BaseManifold, Y, p, q) = Y .= (q - p) / 2
Manifolds.project!(::BaseManifold, Y, p, X) = Y .= 2 .* X
Manifolds.project!(::BaseManifold, q, p) = (q .= p)
Manifolds.injectivity_radius(::BaseManifold) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::Any) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::AbstractRetractionMethod) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::ExponentialRetraction) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::Any, ::AbstractRetractionMethod) = Inf
Manifolds.injectivity_radius(::BaseManifold, ::Any, ::ExponentialRetraction) = Inf
function Manifolds.local_metric(
    ::MetricManifold{ℝ,BaseManifold{N},BaseManifoldMetric{N}},
    A::Manifolds.AbstractAtlas,
    i,
    p,
) where {N}
    return 2 * one(p * p')
end
function Manifolds.exp!(
    M::MetricManifold{ℝ,BaseManifold{N},BaseManifoldMetric{N}},
    q,
    p,
    X,
) where {N}
    return exp!(base_manifold(M), q, p, X)
end
function Manifolds.vector_transport_to!(::BaseManifold, Y, p, X, q, ::ParallelTransport)
    return (Y .= X)
end
function Manifolds.get_basis(
    ::BaseManifold{N},
    p,
    B::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType},
) where {N}
    return CachedBasis(B, [(Matrix{eltype(p)}(I, N, N)[:, i]) for i in 1:N])
end
function Manifolds.get_coordinates!(
    ::BaseManifold,
    Y,
    p,
    X,
    ::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType},
)
    return Y .= X
end
function Manifolds.get_vector!(
    ::BaseManifold,
    Y,
    p,
    X,
    ::DefaultOrthonormalBasis{<:Any,ManifoldsBase.TangentSpaceType},
)
    return Y .= X
end
Manifolds.default_metric_dispatch(::BaseManifold, ::DefaultBaseManifoldMetric) = Val(true)
function Manifolds.projected_distribution(M::BaseManifold, d)
    return ProjectedPointDistribution(M, d, project!, rand(d))
end
function Manifolds.projected_distribution(M::BaseManifold, d, p)
    return ProjectedPointDistribution(M, d, project!, p)
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

        p = [1.0, 2.0, 3.0]
        X = [2.0, 3.0, 4.0]
        @test_throws ErrorException exp(M, p, X)
        using OrdinaryDiffEq
        exp(M, p, X)
    end
    @testset "Local Metric Error message" begin
        M = MetricManifold(BaseManifold{2}(), NotImplementedMetric())
        A = Manifolds.get_default_atlas(M)
        p = [3, 4]
        i = Manifolds.get_chart_index(M, A, p)
        @test_throws ErrorException local_metric(M, A, i, p)
    end
    @testset "scaled Euclidean metric" begin
        n = 3
        E = TestEuclidean{n}()
        g = TestEuclideanMetric()
        M = MetricManifold(E, g)
        A = Manifolds.get_default_atlas(M)
        @test repr(M) == "MetricManifold(TestEuclidean{3}(), TestEuclideanMetric())"

        @test TestEuclideanMetric()(E) === M
        @test TestEuclideanMetric(E) === M

        G = Diagonal(1.0:n)
        invG = inv(G)
        @test manifold_dimension(M) == n
        @test base_manifold(M) === E
        @test metric(M) === g

        i_zeros = Manifolds.get_chart_index(M, A, zeros(3))
        @test_throws ErrorException local_metric_jacobian(E, A, i_zeros, zeros(3))
        @test_throws ErrorException christoffel_symbols_second_jacobian(
            E,
            A,
            i_zeros,
            zeros(3),
        )

        for vtype in (Vector, MVector{n})
            p, X, Y = vtype(randn(n)), vtype(randn(n)), vtype(randn(n))

            chart_p = Manifolds.get_chart_index(M, A, p)

            @test check_manifold_point(M, p) == check_manifold_point(E, p)
            @test check_tangent_vector(M, p, X) == check_tangent_vector(E, p, X)

            @test local_metric(M, A, chart_p, p) ≈ G
            @test inverse_local_metric(M, A, chart_p, p) ≈ invG
            @test det_local_metric(M, A, chart_p, p) ≈ *(1.0:n...)
            @test log_local_metric_density(M, A, chart_p, p) ≈ sum(log.(1.0:n)) / 2
            @test inner(M, p, X, Y) ≈ dot(X, G * Y) atol = 1e-6
            @test norm(M, p, X) ≈ sqrt(dot(X, G * X)) atol = 1e-6

            if VERSION ≥ v"1.1"
                T = 0:0.5:10
                @test geodesic(M, p, X, T) ≈ [p + t * X for t in T] atol = 1e-6
            end

            @test christoffel_symbols_first(M, A, chart_p, p) ≈ zeros(n, n, n) atol = 1e-6
            @test christoffel_symbols_second(M, A, chart_p, p) ≈ zeros(n, n, n) atol = 1e-6
            @test riemann_tensor(M, A, chart_p, p) ≈ zeros(n, n, n, n) atol = 1e-6
            @test ricci_tensor(M, A, chart_p, p) ≈ zeros(n, n) atol = 1e-6
            @test ricci_curvature(M, A, chart_p, p) ≈ 0 atol = 1e-6
            @test gaussian_curvature(M, A, chart_p, p) ≈ 0 atol = 1e-6
            @test einstein_tensor(M, A, chart_p, p) ≈ zeros(n, n) atol = 1e-6

            fdm = FiniteDifferencesBackend(forward_fdm(2, 1))
            @test christoffel_symbols_first(M, A, chart_p, p; backend=fdm) ≈ zeros(n, n, n) atol =
                1e-6
            @test christoffel_symbols_second(M, A, chart_p, p; backend=fdm) ≈ zeros(n, n, n) atol =
                1e-6
            @test riemann_tensor(M, A, chart_p, p; backend=fdm) ≈ zeros(n, n, n, n) atol =
                1e-6
            @test ricci_tensor(M, A, chart_p, p; backend=fdm) ≈ zeros(n, n) atol = 1e-6
            @test ricci_curvature(M, A, chart_p, p; backend=fdm) ≈ 0 atol = 1e-6
            @test gaussian_curvature(M, A, chart_p, p; backend=fdm) ≈ 0 atol = 1e-6
            @test einstein_tensor(M, A, chart_p, p; backend=fdm) ≈ zeros(n, n) atol = 1e-6

            fwd_diff = Manifolds.ForwardDiffBackend()
            @test christoffel_symbols_first(M, A, chart_p, p; backend=fwd_diff) ≈
                  zeros(n, n, n) atol = 1e-6
            @test christoffel_symbols_second(M, A, chart_p, p; backend=fwd_diff) ≈
                  zeros(n, n, n) atol = 1e-6
            @test riemann_tensor(M, A, chart_p, p; backend=fwd_diff) ≈ zeros(n, n, n, n) atol =
                1e-6
            @test ricci_tensor(M, A, chart_p, p; backend=fwd_diff) ≈ zeros(n, n) atol = 1e-6
            @test ricci_curvature(M, A, chart_p, p; backend=fwd_diff) ≈ 0 atol = 1e-6
            @test gaussian_curvature(M, A, chart_p, p; backend=fwd_diff) ≈ 0 atol = 1e-6
            @test einstein_tensor(M, A, chart_p, p; backend=fwd_diff) ≈ zeros(n, n) atol =
                1e-6
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
        A = Manifolds.get_default_atlas(M)

        @test manifold_dimension(M) == n
        @test base_manifold(M) === Sr
        @test metric(M) === g

        for vtype in (Vector, MVector{n})
            p = vtype([θ, ϕ])
            chart_p = Manifolds.get_chart_index(M, A, p)
            G = Diagonal(vtype([1, sin(θ)^2])) .* r^2
            invG = Diagonal(vtype([1, 1 / sin(θ)^2])) ./ r^2
            X, Y = normalize(randn(n)), normalize(randn(n))

            @test local_metric(M, A, chart_p, p) ≈ G atol = 1e-6
            @test inverse_local_metric(M, A, chart_p, p) ≈ invG atol = 1e-6
            @test det_local_metric(M, A, chart_p, p) ≈ r^4 * sin(θ)^2 atol = 1e-6
            @test log_local_metric_density(M, A, chart_p, p) ≈ 2 * log(r) + log(sin(θ)) atol =
                1e-6
            @test inner(M, p, X, Y) ≈ dot(X, G * Y) atol = 1e-6
            @test norm(M, p, X) ≈ sqrt(dot(X, G * X)) atol = 1e-6

            pcart = sph_to_cart(θ, ϕ)
            Xcart = [
                cos(ϕ)*cos(θ) -sin(ϕ)*sin(θ)
                sin(ϕ)*cos(θ) cos(ϕ)*sin(θ)
                -sin(θ) 0
            ] * X

            if VERSION ≥ v"1.1" && (!Sys.iswindows() || Sys.ARCH == :x86_64)
                @testset "numerically integrated geodesics for $vtype" begin
                    T = 0:0.1:1
                    @test isapprox(
                        [sph_to_cart(yi...) for yi in geodesic(M, p, X, T)],
                        geodesic(S, pcart, Xcart, T);
                        atol=1e-3,
                        rtol=1e-3,
                    )
                end
            end

            Γ₁ = christoffel_symbols_first(M, A, chart_p, p)
            for i in 1:n, j in 1:n, k in 1:n
                if (i, j, k) == (1, 2, 2) || (i, j, k) == (2, 1, 2)
                    @test Γ₁[i, j, k] ≈ r^2 * cos(θ) * sin(θ) atol = 1e-6
                elseif (i, j, k) == (2, 2, 1)
                    @test Γ₁[i, j, k] ≈ -r^2 * cos(θ) * sin(θ) atol = 1e-6
                else
                    @test Γ₁[i, j, k] ≈ 0 atol = 1e-6
                end
            end

            Γ₂ = christoffel_symbols_second(M, A, chart_p, p)
            for l in 1:n, i in 1:n, j in 1:n
                if (l, i, j) == (1, 2, 2)
                    @test Γ₂[l, i, j] ≈ -cos(θ) * sin(θ) atol = 1e-6
                elseif (l, i, j) == (2, 1, 2) || (l, i, j) == (2, 2, 1)
                    @test Γ₂[l, i, j] ≈ cot(θ) atol = 1e-6
                else
                    @test Γ₂[l, i, j] ≈ 0 atol = 1e-6
                end
            end

            R = riemann_tensor(M, A, chart_p, p)
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

            @test ricci_tensor(M, A, chart_p, p) ≈ G ./ r^2 atol = 2e-6
            @test ricci_curvature(M, A, chart_p, p) ≈ 2 / r^2 atol = 2e-6
            @test gaussian_curvature(M, A, chart_p, p) ≈ 1 / r^2 atol = 2e-6
            @test einstein_tensor(M, A, chart_p, p) ≈
                  ricci_tensor(M, A, chart_p, p) - gaussian_curvature(M, A, chart_p, p) .* G atol =
                1e-6
        end
    end

    @testset "Metric decorator" begin
        M = BaseManifold{3}()
        g = BaseManifoldMetric{3}()
        MM = MetricManifold(M, g)

        @test DefaultBaseManifoldMetric(BaseManifold{3}()) ===
              MetricManifold(BaseManifold{3}(), DefaultBaseManifoldMetric())
        MT = DefaultBaseManifoldMetric()
        @test MT(BaseManifold{3}()) ===
              MetricManifold(BaseManifold{3}(), DefaultBaseManifoldMetric())

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
        p = [0.1, 0.2, 0.4]
        X = [0.5, 0.7, 0.11]
        Y = [0.13, 0.17, 0.19]
        q = allocate(p)

        @test inner(M, p, X, Y) == 2 * dot(X, Y)
        @test inner(MM, p, X, Y) === inner(M, p, X, Y)
        @test norm(MM, p, X) === norm(M, p, X)
        @test exp(M, p, X) == p + 2 * X
        @test exp(MM2, p, X) == exp(M, p, X)
        @test exp!(MM, q, p, X) === exp!(M, q, p, X)
        @test retract!(MM, q, p, X) === retract!(M, q, p, X)
        @test retract!(MM, q, p, X, 1) === retract!(M, q, p, X, 1)
        # without a definition for the metric from the embedding, no projection possible
        @test_throws ErrorException log!(MM, Y, p, q) === project!(M, Y, p, q)
        @test_throws ErrorException project!(MM, Y, p, X) === project!(M, Y, p, X)
        @test_throws ErrorException project!(MM, q, p) === project!(M, q, p)
        @test_throws ErrorException vector_transport_to!(MM, Y, p, X, q) ===
                                    vector_transport_to!(M, Y, p, X, q)
        # without DiffEq, these error
        # @test_throws ErrorException exp(MM,x, X, 1:3)
        # @test_throws ErrorException exp!(MM, q, p, X)
        # these always fall back anyways.
        @test zero_tangent_vector!(MM, X, p) === zero_tangent_vector!(M, X, p)

        @test injectivity_radius(MM, p) === injectivity_radius(M, p)
        @test injectivity_radius(MM) === injectivity_radius(M)
        @test injectivity_radius(MM, ProjectionRetraction()) ===
              injectivity_radius(M, ProjectionRetraction())
        @test injectivity_radius(MM, ExponentialRetraction()) ===
              injectivity_radius(M, ExponentialRetraction())
        @test injectivity_radius(MM) === injectivity_radius(M)

        @test is_manifold_point(MM, p) === is_manifold_point(M, p)
        @test is_tangent_vector(MM, p, X) === is_tangent_vector(M, p, X)

        A = Manifolds.get_default_atlas(MM2)
        chart_p = Manifolds.get_chart_index(MM2, A, p)
        @test_throws ErrorException local_metric(MM2, A, chart_p, p)
        @test_throws ErrorException local_metric_jacobian(MM2, A, chart_p, p)
        @test_throws ErrorException christoffel_symbols_second_jacobian(MM2, A, chart_p, p)
        # MM falls back to nondefault error
        @test_throws MethodError projected_distribution(MM, 1, p)
        @test_throws MethodError projected_distribution(MM, 1)

        @test inner(MM2, p, X, Y) === inner(M, p, X, Y)
        @test norm(MM2, p, X) === norm(M, p, X)
        @test distance(MM2, p, q) === distance(M, p, q)
        @test exp!(MM2, q, p, X) === exp!(M, q, p, X)
        @test exp(MM2, p, X) == exp(M, p, X)
        @test log!(MM2, X, p, q) === log!(M, X, p, q)
        @test log(MM2, p, q) == log(M, p, q)
        @test retract!(MM2, q, p, X) === retract!(M, q, p, X)
        @test retract!(MM2, q, p, X, 1) === retract!(M, q, p, X, 1)

        @test project!(MM2, q, p) === project!(M, q, p)
        @test project!(MM2, Y, p, X) === project!(M, Y, p, X)
        @test vector_transport_to!(MM2, Y, p, X, q) == vector_transport_to!(M, Y, p, X, q)
        @test zero_tangent_vector!(MM2, X, p) === zero_tangent_vector!(M, X, p)
        @test injectivity_radius(MM2, p) === injectivity_radius(M, p)
        @test injectivity_radius(MM2) === injectivity_radius(M)
        @test injectivity_radius(MM2, p, ExponentialRetraction()) ===
              injectivity_radius(M, p, ExponentialRetraction())
        @test injectivity_radius(MM2, ExponentialRetraction()) ===
              injectivity_radius(M, ExponentialRetraction())
        @test injectivity_radius(MM2, p, ProjectionRetraction()) ===
              injectivity_radius(M, p, ProjectionRetraction())
        @test injectivity_radius(MM2, ProjectionRetraction()) ===
              injectivity_radius(M, ProjectionRetraction())
        @test is_manifold_point(MM2, p) === is_manifold_point(M, p)
        @test is_tangent_vector(MM2, p, X) === is_tangent_vector(M, p, X)

        a = Manifolds.projected_distribution(M, Distributions.MvNormal(zero(zeros(3)), 1.0))
        b = Manifolds.projected_distribution(
            MM2,
            Distributions.MvNormal(zero(zeros(3)), 1.0),
        )
        @test isapprox(Matrix(a.distribution.Σ), Matrix(b.distribution.Σ))
        @test isapprox(a.distribution.μ, b.distribution.μ)
        @test get_basis(M, p, DefaultOrthonormalBasis()).data ==
              get_basis(MM2, p, DefaultOrthonormalBasis()).data
        @test_throws ErrorException get_basis(MM, p, DefaultOrthonormalBasis())
        coX = flat(M, p, X)
        coY = flat(M, p, Y)
        @test coX(X) ≈ norm(M, p, X)^2
        @test coY(X) ≈ inner(M, p, X, Y)
        cotspace = CotangentBundleFibers(M)
        cotspace2 = CotangentBundleFibers(MM)
        @test coX.X ≈ X
        @test inner(M, p, X, Y) ≈ inner(cotspace, p, coX, coY)
        @test inner(MM, p, X, Y) ≈ inner(cotspace, p, coX, coY)
        @test inner(MM, p, X, Y) ≈ inner(cotspace2, p, coX, coY)
        @test sharp(M, p, coX) ≈ X

        psample = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        Y = pweights([0.5, 0.5])
        # test despatch with results from above
        @test mean(M, psample, Y) ≈ ones(3)
        @test mean(MM2, psample, Y) ≈ ones(3)
        @test mean(MM, psample, Y) ≈ 3 .* ones(3)

        @test median(M, psample, Y) ≈ 2 .* ones(3)
        @test median(MM2, psample, Y) ≈ 2 * ones(3)
        @test median(MM, psample, Y) ≈ 4 .* ones(3)
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
