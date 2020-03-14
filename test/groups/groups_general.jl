
include("../utils.jl")
include("group_utils.jl")

@testset "General group tests" begin
    @test length(methods(has_biinvariant_metric)) == 1
    @test length(methods(has_invariant_metric)) == 1
    @test length(methods(has_biinvariant_metric)) == 1
    @testset "Not implemented operation" begin
        G = GroupManifold(NotImplementedManifold(), NotImplementedOperation())
        @test repr(G) == "GroupManifold(NotImplementedManifold(), NotImplementedOperation())"
        x = [1.0, 2.0]
        v = [2.0, 3.0]
        eg = Identity(G, [0.0, 0.0])
        @test repr(eg) === "Identity($(G), $([0.0, 0.0]))"
        @test length(methods(is_group_decorator)) == 1

        @test Manifolds.is_group_decorator(G)
        @test Manifolds.decorator_group_dispatch(G) === Val{true}()
        @test Manifolds.default_decorator_dispatch(G) === Val{false}()
        @test !Manifolds.is_group_decorator(NotImplementedManifold())
        @test Manifolds.decorator_group_dispatch(NotImplementedManifold()) === Val{false}()

        @test Manifolds.decorator_transparent_dispatch(compose, G, x, x, x) === Val{:intransparent}()
        @test Manifolds.decorator_transparent_dispatch(compose!, G, x, x, x) === Val{:intransparent}()
        @test Manifolds.decorator_transparent_dispatch(group_exp, G, x, x) === Val{:intransparent}()
        @test Manifolds.decorator_transparent_dispatch(group_log, G, x, x) === Val{:intransparent}()
        @test Manifolds.decorator_transparent_dispatch(translate_diff!, G, x, x, x, x, x) === Val{:intransparent}()
        @test base_group(G) === G

        if VERSION ≥ v"1.3"
            @test NotImplementedOperation(NotImplementedManifold()) === G
            @test (NotImplementedOperation())(NotImplementedManifold()) === G
        end
        @test_throws ErrorException base_group(MetricManifold(Euclidean(3), EuclideanMetric()))
        @test_throws ErrorException hat(Rotations(3), eg, [1, 2, 3])
        @test_throws ErrorException hat(GroupManifold(Rotations(3), NotImplementedOperation()), eg, [1, 2, 3])
        @test_throws ErrorException vee(Rotations(3), eg, [1, 2, 3])
        @test_throws ErrorException vee(GroupManifold(Rotations(3), NotImplementedOperation()), eg, [1, 2, 3])
        @test_throws ErrorException Identity(Euclidean(3), [0, 0, 0])

        @test_throws ErrorException inv!(G, x, x)
        @test_throws ErrorException inv!(G, x, eg)
        @test_throws ErrorException inv(G, x)

        @test copyto!(x, eg) === x
        @test isapprox(G, x, eg)
        @test_throws ErrorException identity!(G, x, x)
        @test_throws ErrorException identity(G, x)

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

        for f in [compose, compose!, translate_diff!, translate_diff]
            @test Manifolds.decorator_transparent_dispatch(f, G) === Val{:transparent}()
        end
        for f in [translate, translate!]
            @test Manifolds.decorator_transparent_dispatch(f, G) === Val{:intransparent}()
        end
        for f in [inverse_translate_diff!, inverse_translate_diff]
            @test Manifolds.decorator_transparent_dispatch(f, G) === Val{:transparent}()
        end
        for f in [group_exp!, group_exp, group_log, group_log!]
            @test Manifolds.decorator_transparent_dispatch(f, G) === Val{:transparent}()
        end
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

        @test_throws DomainError is_manifold_point(
            G,
            Identity(GroupManifold(NotImplementedManifold(), NotImplementedOperation()), [0.0, 0.0]),
            true,
        )

        x = [1.0, 2.0]
        v = [3.0, 4.0]
        ge = Identity(G, [0.0, 0.0])
        @test zero(ge) === ge
        @test number_eltype(ge) == Bool
        @test copyto!(ge, ge) === ge
        y = allocate(x)
        copyto!(y, ge)
        @test y ≈ zero(x)
        @test ge - x == -x
        @test x - ge === x
        @test ge - ge === ge
        @test ge + x ≈ x
        @test x + ge ≈ x
        @test ge + ge === ge
        @test -ge === ge
        @test +ge === ge
        @test ge * 1 === ge
        @test 1 * ge === ge
        @test ge * ge === ge
        @test ge.p ≈ zero(x)
        @test inv(G, x) ≈ -x
        @test inv(G, ge) === ge
        @test identity(G, x) ≈ zero(x)
        @test identity(G, ge) === ge
        y = allocate(x)
        identity!(G, y, x)
        @test y ≈ zero(x)
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

        y = identity(G, x)
        @test isapprox(y, ge; atol=1e-10)
        @test isapprox(ge, y; atol=1e-10)
        @test isapprox(ge, ge)
    end

    @testset "Multiplication operation" begin
        G = GroupManifold(NotImplementedManifold(), Manifolds.MultiplicationOperation())
        test_group(
            G,
            [[1.0 2.0; 3.0 4.0], [2.0 3.0; 4.0 5.0], [3.0 4.0; 5.0 6.0]],
            [],
            [[1.0 2.0; 3.0 4.0]];
            test_group_exp_log = false,
        )

        x = [1.0 2.0; 2.0 3.0]
        ge = Identity(G, [1.0 0.0; 0.0 1.0])
        @test number_eltype(ge) == Bool
        @test copyto!(ge, ge) === ge
        y = allocate(x)
        copyto!(y, ge)
        @test y ≈ one(x)
        @test one(ge) === ge
        @test transpose(ge) === ge
        @test det(ge) == 1
        @test ge * x ≈ x
        @test x * ge ≈ x
        @test ge * ge === ge
        @test inv(ge) === ge
        @test *(ge) === ge

        @test x / ge ≈ x
        @test ge \ x ≈ x
        @test ge / ge === ge
        @test ge \ ge === ge
        @test ge / x ≈ inv(x)
        @test x \ ge ≈ inv(x)
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
        @test identity(G, x) ≈ one(x)
        @test identity(G, ge) === ge
        y = allocate(x)
        identity!(G, y, x)
        @test y ≈ one(x)
        @test_throws ErrorException identity!(G, [0.0], ge)
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
        @test group_exp!(G, y, x) === y
        @test y ≈ exp(x)

        @testset "identity optimization" begin
            x2 = copy(x)
            identity!(G, x2, x)
            x3 = copy(x)
            invoke(identity!, Tuple{AbstractGroupManifold{Manifolds.MultiplicationOperation}, Any, AbstractMatrix}, G, x3, x)
            @test isapprox(G, x2, x3)
        end
    end
end

struct NotImplementedAction <: AbstractGroupAction{LeftAction}
end

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
