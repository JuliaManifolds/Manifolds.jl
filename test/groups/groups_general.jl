using StaticArrays: identity_perm
using Manifolds: decorator_transparent_dispatch
using Base: decode_overlong

include("../utils.jl")
include("group_utils.jl")

@testset "General group tests" begin
    @test length(methods(has_biinvariant_metric)) == 1
    @test length(methods(has_invariant_metric)) == 1
    @test length(methods(has_biinvariant_metric)) == 1
    @testset "Not implemented operation" begin
        G = GroupManifold(NotImplementedManifold(), NotImplementedOperation())
        @test repr(G) ==
              "GroupManifold(NotImplementedManifold(), NotImplementedOperation())"
        x = [1.0, 2.0]
        v = [2.0, 3.0]
        eg = Identity(G)
        @test repr(eg) === "Identity(NotImplementedOperation)"
        @test number_eltype(eg) == Bool
        @test is_identity(G, eg) # identity transparent
        p = similar(x)
        copyto!(G, p, e)
        @test p == identity_element(G)
        @test isapprox(G, eg, p)
        @test isapprox(G, p, eg)
        @test isapprox(G, eg, eg)
        @test length(methods(is_group_decorator)) == 1

        @test Manifolds.is_group_decorator(G)
        @test Manifolds.decorator_group_dispatch(G) === Val{true}()
        @test Manifolds.default_decorator_dispatch(G) === Val{false}()
        @test !Manifolds.is_group_decorator(NotImplementedManifold())
        @test Manifolds.decorator_group_dispatch(NotImplementedManifold()) === Val{false}()

        @test Manifolds.decorator_transparent_dispatch(compose, G, x, x, x) ===
              Val{:intransparent}()
        @test Manifolds.decorator_transparent_dispatch(compose!, G, x, x, x) ===
              Val{:intransparent}()
        @test Manifolds.decorator_transparent_dispatch(group_exp, G, x, x) ===
              Val{:intransparent}()
        @test Manifolds.decorator_transparent_dispatch(group_log, G, x, x) ===
              Val{:intransparent}()
        @test Manifolds.decorator_transparent_dispatch(
            translate_diff!,
            G,
            x,
            x,
            x,
            x,
            x,
        ) === Val{:intransparent}()
        @test base_group(G) === G
        z = similar(x)
        copyto!(G, z, eg)
        @test z == eg.p
        @test NotImplementedOperation(NotImplementedManifold()) === G
        @test (NotImplementedOperation())(NotImplementedManifold()) === G

        @test_throws ErrorException base_group(
            MetricManifold(Euclidean(3), EuclideanMetric()),
        )
        @test_throws ErrorException hat(Rotations(3), eg, [1, 2, 3])
        @test_throws ErrorException hat(
            GroupManifold(Rotations(3), NotImplementedOperation()),
            eg,
            [1, 2, 3],
        )
        @test_throws ErrorException vee(Rotations(3), eg, [1, 2, 3])
        @test_throws ErrorException vee(
            GroupManifold(Rotations(3), NotImplementedOperation()),
            eg,
            [1, 2, 3],
        )

        @test_throws ErrorException inv!(G, x, x)
        @test_throws ErrorException inv!(G, x, eg)
        @test_throws ErrorException inv(G, x)

        @test copyto!(G, x, eg) === x
        @test isapprox(G, x, eg)

        @test_throws ErrorException compose(G, x, x)
        @test_throws ErrorException compose(G, x, eg)
        @test_throws ErrorException compose!(G, x, eg, x)
        @test_throws ErrorException compose!(G, x, x, eg)
        @test_throws ErrorException compose!(G, x, x, x)
        @test_throws ErrorException compose!(G, x, eg, eg)

        @test_throws ErrorException translate(G, x, x)
        @test_throws ErrorException translate(G, x, x, LeftAction())
        @test_throws ErrorException translate(G, x, x, RightAction())
        @test_throws ErrorException translate!(G, x, x, x)
        @test_throws ErrorException translate!(G, x, x, x, LeftAction())
        @test_throws ErrorException translate!(G, x, x, x, RightAction())

        @test_throws ErrorException inverse_translate(G, x, x)
        @test_throws ErrorException inverse_translate(G, x, x, LeftAction())
        @test_throws ErrorException inverse_translate(G, x, x, RightAction())
        @test_throws ErrorException inverse_translate!(G, x, x, x)
        @test_throws ErrorException inverse_translate!(G, x, x, x, LeftAction())
        @test_throws ErrorException inverse_translate!(G, x, x, x, RightAction())

        @test_throws ErrorException translate_diff(G, x, x, v)
        @test_throws ErrorException translate_diff(G, x, x, v, LeftAction())
        @test_throws ErrorException translate_diff(G, x, x, v, RightAction())
        @test_throws ErrorException translate_diff!(G, v, x, x, v)
        @test_throws ErrorException translate_diff!(G, v, x, x, v, LeftAction())
        @test_throws ErrorException translate_diff!(G, v, x, x, v, RightAction())

        @test_throws ErrorException inverse_translate_diff(G, x, x, v)
        @test_throws ErrorException inverse_translate_diff(G, x, x, v, LeftAction())
        @test_throws ErrorException inverse_translate_diff(G, x, x, v, RightAction())
        @test_throws ErrorException inverse_translate_diff!(G, v, x, x, v)
        @test_throws ErrorException inverse_translate_diff!(G, v, x, x, v, LeftAction())
        @test_throws ErrorException inverse_translate_diff!(G, v, x, x, v, RightAction())

        @test_throws ErrorException group_exp(G, v)
        @test_throws ErrorException group_exp!(G, x, v)
        @test_throws ErrorException group_log(G, x)
        @test_throws ErrorException group_log!(G, v, x)

        for f in [translate, translate!]
            @test Manifolds.decorator_transparent_dispatch(f, G) === Val{:intransparent}()
        end
        for f in [inverse_translate_diff!, inverse_translate_diff]
            @test Manifolds.decorator_transparent_dispatch(f, G) === Val{:transparent}()
        end
        for f in [group_exp!, group_exp, group_log, group_log!]
            @test Manifolds.decorator_transparent_dispatch(f, G, x, x) ===
                  Val{:intransparent}()
        end
        for f in [get_vector, get_coordinates]
            @test Manifolds.decorator_transparent_dispatch(f, G) === Val{:parent}()
        end
        @test Manifolds.decorator_transparent_dispatch(isapprox, G, eg, x) ===
              Val{:transparent}()
        @test Manifolds.decorator_transparent_dispatch(isapprox, G, x, eg) ===
              Val{:transparent}()
        @test Manifolds.decorator_transparent_dispatch(isapprox, G, eg, eg) ===
              Val{:transparent}()
    end

    @testset "Action direction" begin
        @test switch_direction(LeftAction()) == RightAction()
        @test switch_direction(RightAction()) == LeftAction()

        @test Manifolds._action_order(1, 2, LeftAction()) === (1, 2)
        @test Manifolds._action_order(1, 2, RightAction()) === (2, 1)
    end

    @testset "Addition operation" begin
        G = GroupManifold(NotImplementedManifold(), Manifolds.AdditionOperation())
        test_group(G, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [], [[1.0, 2.0]])

        @test_throws DomainError is_point(
            G,
            Identity(
                GroupManifold(NotImplementedManifold(), NotImplementedOperation()),
            ),
            true,
        )

        x = [1.0, 2.0]
        v = [3.0, 4.0]
        ge = Identity(G)
        @test number_eltype(ge) == Bool
        y = allocate(x)
        copyto!(G, y, ge)
        @test y ≈ zero(x)
        @test ge - x == -x
        @test x - ge === x
        @test ge - ge === ge
        @test ge + x ≈ x
        @test x + ge ≈ x
        @test ge + ge === ge
        @test -(ge) === ge
        @test +(ge) === ge
        @test ge * 1 === ge
        @test 1 * ge === ge
        @test ge * ge === ge
        @test ge.p ≈ zero(x)
        @test zero(ge) == ge
        @test inv(G, x) ≈ -x
        @test inv(G, ge) === ge
        @test compose(G, x, x) ≈ x + x
        @test compose(G, x, ge) ≈ x
        @test compose(G, ge, x) ≈ x
        @test compose(G, ge, ge) == ge
        compose!(G, y, x, x)
        @test y ≈ x + x
        compose!(G, y, x, ge)
        @test y ≈ x
        compose!(G, y, ge, x)
        @test y ≈ x
        @test group_exp(G, v) === v
        @test group_log(G, x) === x
    end

    @testset "Multiplication operation" begin
        G = GroupManifold(NotImplementedManifold(), Manifolds.MultiplicationOperation())
        test_group(
            G,
            [[2.0 1.0; 3.0 4.0], [3.0 2.0; 4.0 5.0], [4.0 3.0; 5.0 6.0]],
            [],
            [[1.0 2.0; 3.0 4.0]];
            test_group_exp_log=true,
        )

        x = [2.0 1.0; 2.0 3.0]
        ge = Identity(G)
        @test number_eltype(ge) == Bool
        @test copyto!(ge, ge) === ge
        y = allocate(x)
        identity_element!(G, y)
        @test y ≈ one(x)
        @test one(ge) === ge
        @test transpose(ge) === ge
        @test det(ge) == 1
        @test ge * x ≈ x
        @test x * ge ≈ x
        @test ge * ge === ge
        @test inv(G, ge) === ge
        @test *(ge) === ge

        @test x / ge ≈ x
        @test ge \ x ≈ x
        @test ge / ge === ge
        @test ge \ ge === ge
        @test ge / x ≈ inv(G, x)
        @test x \ ge ≈ inv(G, x)
        y = allocate(x)
        @test LinearAlgebra.mul!(y, x, ge) === y
        @test y ≈ x
        y = allocate(x)
        @test LinearAlgebra.mul!(y, ge, x) === y
        @test y ≈ x
        y = allocate(x)
        @test LinearAlgebra.mul!(y, ge, ge) === y
        @test y ≈ one(y)

        @test ge.p ≈ one(x)
        @test inv(G, x) ≈ inv(x)
        @test inv(G, ge) === ge
        z = allocate(x)
        copyto!(G, z, x)
        z2 = allocate(x)
        copyto!(G.manifold, z2, x)
        @test z == z2
        X = zeros(2, 2)
        Y = allocate(X)
        copyto!(G, Y, x, X)
        Y2 = allocate(X)
        copyto!(G.manifold, Y2, x, X)
        @test Y == Y2

        @test compose(G, x, x) ≈ x * x
        @test compose(G, x, ge) ≈ x
        @test compose(G, ge, x) ≈ x
        @test compose(G, ge, ge) == ge
        compose!(G, y, x, x)
        @test y ≈ x * x
        compose!(G, y, x, ge)
        @test y ≈ x
        compose!(G, y, ge, x)
        @test y ≈ x
        X = [1.0 2.0; 3.0 4.0]
        @test group_exp!(G, y, X) === y
        @test_throws ErrorException group_exp!(G, y, :a)
        @test y ≈ exp(X)
        Y = allocate(X)
        @test group_log!(G, Y, y) === Y
        @test Y ≈ log(y)
    end

    @testset "Identity on Group Manifolds" begin
        G = TranslationGroup(3)
        e = Identity(G)
        @test get_vector(G, e, ones(3), DefaultOrthogonalBasis()) == ones(3)
        @test e - e == e
        @test ones(3) + e == ones(3)
    end

    @testset "Transparency tests" begin
        G = DefaultTransparencyGroup(Euclidean(3), AdditionOperation())
        p = ones(3)
        q = 2 * p
        X = zeros(3)
        Y = similar(X)
        for f in
            [vector_transport_along!, vector_transport_direction!, vector_transport_to!]
            @test ManifoldsBase.decorator_transparent_dispatch(
                f,
                G,
                Y,
                p,
                X,
                q,
                ParallelTransport(),
            ) == Val(:intransparent)
        end
    end
end

struct NotImplementedAction <: AbstractGroupAction{LeftAction} end

@testset "General group action tests" begin
    @testset "Not implemented operations" begin
        A = NotImplementedAction()
        x = [1.0, 2.0]
        a = [1.0, 2.0]
        v = [1.0, 2.0]

        @test_throws ErrorException base_group(A)
        @test_throws ErrorException g_manifold(A)
        @test_throws ErrorException apply(A, a, x)
        @test_throws ErrorException apply!(A, x, a, x)
        @test_throws ErrorException inverse_apply(A, a, x)
        @test_throws ErrorException inverse_apply!(A, x, a, x)
        @test_throws ErrorException apply_diff(A, a, x, v)
        @test_throws ErrorException apply_diff!(A, v, x, a, v)
        @test_throws ErrorException inverse_apply_diff(A, a, x, v)
        @test_throws ErrorException inverse_apply_diff!(A, v, x, a, v)
        @test_throws ErrorException compose(A, a, a)
        @test_throws ErrorException compose!(A, a, a, a)
        @test_throws ErrorException optimal_alignment(A, x, x)
        @test_throws ErrorException optimal_alignment!(A, a, x, x)
    end
end
