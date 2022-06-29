include("../utils.jl")
include("group_utils.jl")

using OrdinaryDiffEq
import Manifolds: local_metric

using Manifolds: LeftInvariantMetric, RightInvariantMetric

struct TestInvariantMetricBase <: AbstractMetric end

function active_traits(
    f,
    M::MetricManifold{𝔽,<:AbstractManifold,TestInvariantMetricBase},
    args...,
) where {𝔽}
    return merge_traits(
        HasLeftInvariantMetric(),
        IsMetricManifold(),
        active_traits(f, M.manifold, args...),
    )
end
function local_metric(
    ::MetricManifold{𝔽,<:AbstractManifold,TestInvariantMetricBase},
    ::Identity,
    ::DefaultOrthonormalBasis,
) where {𝔽}
    return Diagonal([1.0, 2.0, 3.0])
end

struct TestBiInvariantMetricBase <: AbstractMetric end

function active_traits(
    f,
    M::MetricManifold{𝔽,<:AbstractManifold,TestBiInvariantMetricBase},
    args...,
) where {𝔽}
    return merge_traits(
        HasBiinvariantMetric(),
        IsMetricManifold(),
        active_traits(f, M.manifold, args...),
    )
end
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
        Y = similar(X)
        p2 = similar(p)

        G = MetricManifold(SO3, TestBiInvariantMetricBase())
        @test isapprox(SO3, exp(G, p, X), exp(SO3, p, X))
        exp!(G, p2, p, X)
        @test isapprox(SO3, p2, exp(SO3, p, X))
        @test isapprox(SO3, p, log(G, p, q), log(SO3, p, q); atol=1e-6)
        log!(G, Y, p, q)
        @test isapprox(SO3, p, Y, log(SO3, p, q); atol=1e-6)

        G = MetricManifold(SO3, TestBiInvariantMetricBase())
        @test isapprox(SO3, exp(G, p, X), exp(SO3, p, X))
        @test isapprox(SO3, p, log(G, p, q), log(SO3, p, q); atol=1e-6)

        @test is_group_manifold(G)
        @test is_group_manifold(G, MultiplicationOperation())
        @test !isapprox(G, e, Identity(AdditionOperation()))
        @test has_biinvariant_metric(G)
        @test !has_biinvariant_metric(Sphere(2))
    end

    @testset "invariant metric direction" begin
        @test direction(HasRightInvariantMetric()) === RightAction()
        @test direction(HasLeftInvariantMetric()) === LeftAction()
        @test direction(HasRightInvariantMetric) === RightAction()
        @test direction(HasLeftInvariantMetric) === LeftAction()
    end

    @testset "invariant metrics on SE(3)" begin
        basis_types = (DefaultOrthonormalBasis(),)
        for inv_metric in [LeftInvariantMetric(), RightInvariantMetric()]
            G = MetricManifold(SpecialEuclidean(3), inv_metric)

            M = base_manifold(G)
            Rn = Rotations(3)
            p = Matrix(I, 3, 3)

            t = Vector{Float64}.([1:3, 2:4, 4:6])
            ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
            tuple_pts = [(ti, exp(Rn, p, hat(Rn, p, ωi))) for (ti, ωi) in zip(t, ω)]
            tuple_X = [
                ([-1.0, 2.0, 1.0], hat(Rn, p, [1.0, 0.5, -0.5])),
                ([-2.0, 1.0, 0.5], hat(Rn, p, [-1.0, -0.5, 1.1])),
            ]

            pts = [ProductRepr(tp...) for tp in tuple_pts]
            X_pts = [ProductRepr(tX...) for tX in tuple_X]

            g1, g2 = pts[1:2]
            t1, R1 = g1.parts
            t2, R2 = g2.parts
            g1g2 = ProductRepr(R1 * t2 + t1, R1 * R2)
            @test isapprox(G, compose(G, g1, g2), g1g2)

            test_group(
                G,
                pts,
                X_pts,
                X_pts;
                test_diff=true,
                test_lie_bracket=true,
                test_adjoint_action=true,
                diff_convs=[(), (LeftAction(),), (RightAction(),)],
            )
            test_manifold(
                G,
                pts;
                #basis_types_vecs=basis_types,
                basis_types_to_from=basis_types,
                is_mutating=true,
                #test_inplace=true,
                test_vee_hat=false,
                exp_log_atol_multiplier=50,
            )
        end
    end
end
