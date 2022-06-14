using StaticArrays: identity_perm
using Base: decode_overlong

include("../utils.jl")
include("group_utils.jl")

@testset "General group tests" begin
    @testset "Not implemented operation" begin
        G = GroupManifold(NotImplementedManifold(), NotImplementedOperation())
        @test repr(G) ==
              "GroupManifold(NotImplementedManifold(), NotImplementedOperation())"
        p = [1.0, 2.0]
        X = [2.0, 3.0]
        eg = Identity(G)
        @test repr(eg) === "Identity(NotImplementedOperation)"
        @test number_eltype(eg) == Bool
        @test !is_group_manifold(NotImplementedManifold(), NotImplementedOperation())
        @test is_identity(G, eg) # identity transparent
        @test_throws MethodError identity_element(G) # but for a NotImplOp there is no concrete id.
        @test isapprox(G, eg, eg)
        @test !isapprox(G, Identity(AdditionOperation()), eg)
        @test !isapprox(G, Identity(AdditionOperation()), eg)
        @test !isapprox(
            G,
            Identity(AdditionOperation()),
            Identity(MultiplicationOperation()),
        )
        @test_throws DomainError is_point(G, Identity(AdditionOperation()), true)
        @test is_point(G, eg)
        @test_throws MethodError is_identity(G, 1) # same error as before i.e. dispatch isapprox works
        @test Manifolds.check_size(G, eg) === nothing
        @test Manifolds.check_size(
            Manifolds.EmptyTrait(),
            MetricManifold(NotImplementedManifold(), EuclideanMetric()),
            eg,
        ) isa DomainError

        @test Identity(NotImplementedOperation()) === eg
        @test Identity(NotImplementedOperation) === eg
        @test !is_point(G, Identity(AdditionOperation()))
        @test !isapprox(G, eg, Identity(AdditionOperation()))
        @test !isapprox(G, Identity(AdditionOperation()), eg)

        @test NotImplementedOperation(NotImplementedManifold()) === G
        @test (NotImplementedOperation())(NotImplementedManifold()) === G

        @test_throws ErrorException hat(Rotations(3), eg, [1, 2, 3])
        @test_throws ErrorException hat!(Rotations(3), randn(3, 3), eg, [1, 2, 3])
        # If you force it, you get a not that readable MethodError
        @test_throws MethodError hat(
            GroupManifold(Rotations(3), NotImplementedOperation()),
            eg,
            [1, 2, 3],
        )

        @test_throws ErrorException vee(Rotations(3), eg, [1, 2, 3])
        @test_throws ErrorException vee!(Rotations(3), randn(3), eg, [1, 2, 3])
        @test_throws MethodError vee(
            GroupManifold(Rotations(3), NotImplementedOperation()),
            eg,
            [1, 2, 3],
        )

        @test_throws ErrorException inv!(G, p, p)
        @test_throws MethodError inv!(G, p, eg)
        @test_throws ErrorException inv(G, p)

        # no function defined to return the identity array representation
        @test_throws MethodError copyto!(G, p, eg)

        @test_throws MethodError compose(G, p, p)
        @test compose(G, p, eg) == p
        xO = deepcopy(p)
        compose!(G, p, eg, p)
        @test xO == p
        compose!(G, p, p, eg)
        @test xO == p
        @test_throws MethodError compose!(G, p, p, p)
        @test_throws MethodError compose!(G, p, eg, eg)

        @test_throws MethodError translate(G, p, p)
        @test_throws MethodError translate(G, p, p, LeftAction())
        @test_throws MethodError translate(G, p, p, RightAction())
        @test_throws MethodError translate!(G, p, p, p)
        @test_throws MethodError translate!(G, p, p, p, LeftAction())
        @test_throws MethodError translate!(G, p, p, p, RightAction())

        @test_throws ErrorException inverse_translate(G, p, p)
        @test_throws ErrorException inverse_translate(G, p, p, LeftAction())
        @test_throws ErrorException inverse_translate(G, p, p, RightAction())
        @test_throws ErrorException inverse_translate!(G, p, p, p)
        @test_throws ErrorException inverse_translate!(G, p, p, p, LeftAction())
        @test_throws ErrorException inverse_translate!(G, p, p, p, RightAction())

        @test_throws MethodError translate_diff(G, p, p, X)
        @test_throws MethodError translate_diff(G, p, p, X, LeftAction())
        @test_throws MethodError translate_diff(G, p, p, X, RightAction())
        @test_throws MethodError translate_diff!(G, X, p, p, X)
        @test_throws MethodError translate_diff!(G, X, p, p, X, LeftAction())
        @test_throws MethodError translate_diff!(G, X, p, p, X, RightAction())

        @test_throws ErrorException inverse_translate_diff(G, p, p, X)
        @test_throws ErrorException inverse_translate_diff(G, p, p, X, LeftAction())
        @test_throws ErrorException inverse_translate_diff(G, p, p, X, RightAction())
        @test_throws ErrorException inverse_translate_diff!(G, X, p, p, X)
        @test_throws ErrorException inverse_translate_diff!(G, X, p, p, X, LeftAction())
        @test_throws ErrorException inverse_translate_diff!(G, X, p, p, X, RightAction())

        @test_throws MethodError exp_lie(G, X)
        @test_throws MethodError exp_lie!(G, p, X)
        # no transparency error, but _log_lie missing
        @test_throws MethodError log_lie(G, p)
        @test_throws MethodError log_lie!(G, X, p)
    end

    @testset "Action direction" begin
        @test switch_direction(LeftAction()) == RightAction()
        @test switch_direction(RightAction()) == LeftAction()

        @test Manifolds._action_order(1, 2, LeftAction()) === (1, 2)
        @test Manifolds._action_order(1, 2, RightAction()) === (2, 1)
    end

    @testset "Addition operation" begin
        G = GroupManifold(NotImplementedManifold(), Manifolds.AdditionOperation())
        test_group(
            G,
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [],
            [[1.0, 2.0]];
            test_exp_lie_log=false, # there is no identity element so log/exp on Lie do not work
            test_one_arg_identity_element=false,
        )

        p = [1.0, 2.0]
        X = [3.0, 4.0]
        ge = Identity(G)
        @test number_eltype(ge) == Bool
        y = allocate(p)
        copyto!(G, y, ge)
        @test y ≈ zero(p)
        @test ge - p == -p
        @test p - ge === p
        @test ge - ge === ge
        @test ge + p ≈ p
        @test p + ge ≈ p
        @test ge + ge === ge
        @test -(ge) === ge
        @test +(ge) === ge
        @test ge * 1 === ge
        @test 1 * ge === ge
        @test ge * ge === ge
        @test inv(G, p) ≈ -p
        @test inv(G, ge) === ge
        @test compose(G, p, p) ≈ p + p
        @test compose(G, p, ge) ≈ p
        @test compose(G, ge, p) ≈ p
        @test compose(G, ge, ge) == ge
        compose!(G, y, p, p)
        @test y ≈ p + p
        compose!(G, y, p, ge)
        @test y ≈ p
        compose!(G, y, ge, p)
        @test y ≈ p
        @test exp_lie(G, X) === X
        @test log_lie(G, p) === p
    end

    @testset "Multiplication operation" begin
        G = GroupManifold(NotImplementedManifold(), Manifolds.MultiplicationOperation())
        test_group(
            G,
            [[2.0 1.0; 3.0 4.0], [3.0 2.0; 4.0 5.0], [4.0 3.0; 5.0 6.0]],
            [],
            [[1.0 2.0; 3.0 4.0]];
            test_exp_lie_log=false, # no identity available as array
            test_one_arg_identity_element=false,
        )

        p = [2.0 1.0; 2.0 3.0]
        ge = Identity(G)
        @test number_eltype(ge) == Bool
        @test copyto!(G, ge, ge) === ge
        y = allocate(p)
        identity_element!(G, y)
        @test_throws DimensionMismatch identity_element!(G, [1, 2, 3])
        @test y ≈ one(p)
        @test one(ge) === ge
        @test transpose(ge) === ge
        @test det(ge) == 1
        @test ge * p ≈ p
        @test p * ge ≈ p
        @test ge * ge === ge
        @test inv(G, ge) === ge
        @test *(ge) === ge

        @test p / ge ≈ p
        @test ge \ p ≈ p
        @test ge / ge === ge
        @test ge \ ge === ge
        @test ge / p ≈ inv(G, p)
        @test p \ ge ≈ inv(G, p)
        y = allocate(p)
        @test LinearAlgebra.mul!(y, p, ge) === y
        @test y ≈ p
        y = allocate(p)
        @test LinearAlgebra.mul!(y, ge, p) === y
        @test y ≈ p
        y = allocate(p)
        @test LinearAlgebra.mul!(y, ge, ge) === y
        @test y ≈ one(y)

        @test inv(G, p) ≈ inv(p)
        @test inv(G, ge) === ge
        z = allocate(p)
        copyto!(G, z, p)
        z2 = allocate(p)
        copyto!(G.manifold, z2, p)
        @test z == z2
        X = zeros(2, 2)
        Y = allocate(X)
        copyto!(G, Y, p, X)
        Y2 = allocate(X)
        copyto!(G.manifold, Y2, p, X)
        @test Y == Y2

        @test compose(G, p, p) ≈ p * p
        @test compose(G, p, ge) ≈ p
        @test compose(G, ge, p) ≈ p
        @test compose(G, ge, ge) == ge
        compose!(G, y, p, p)
        @test y ≈ p * p
        compose!(G, y, p, ge)
        @test y ≈ p
        compose!(G, y, ge, p)
        @test y ≈ p
        X = [1.0 2.0; 3.0 4.0]
        @test exp_lie!(G, y, X) === y
        @test_throws MethodError exp_lie!(G, y, :a)
        @test y ≈ exp(X)
        Y = allocate(X)
        @test log_lie!(G, Y, y) === Y
        @test Y ≈ log(y)

        q2 = SVDMPoint(2 * Matrix{Float64}(I, 3, 3))
        mul!(q2, ge, ge)
        qT = SVDMPoint(Matrix{Float64}(I, 3, 3))
        @test isapprox(FixedRankMatrices(3, 3, 3), q2, qT)
    end

    @testset "Identity on Group Manifolds" begin
        G = TranslationGroup(3)
        e = Identity(G)
        @test get_vector_lie(G, ones(3), DefaultOrthogonalBasis()) == ones(3)
        @test e - e == e
        @test ones(3) + e == ones(3)
        e_add = Identity(AdditionOperation)
        e_mul = Identity(MultiplicationOperation)
        @test e_add * e_mul === e_add
        @test e_mul * e_add === e_add
        @test mul!(e_mul, e_mul, e_mul) === e_mul
    end
end

struct NotImplementedAction <: AbstractGroupAction{LeftAction} end

@testset "General group action tests" begin
    @testset "Not implemented operations" begin
        A = NotImplementedAction()
        p = [1.0, 2.0]
        a = [1.0, 2.0]
        X = [1.0, 2.0]

        @test_throws ErrorException apply(A, a, p)
        @test_throws ErrorException apply!(A, p, a, p)
        @test_throws ErrorException inverse_apply(A, a, p)
        @test_throws ErrorException inverse_apply!(A, p, a, p)
        @test_throws ErrorException apply_diff(A, a, p, X)
        @test_throws ErrorException apply_diff!(A, X, p, a, X)
        @test_throws ErrorException inverse_apply_diff(A, a, p, X)
        @test_throws ErrorException inverse_apply_diff!(A, X, p, a, X)
        @test_throws ErrorException compose(A, a, a)
        @test_throws ErrorException compose!(A, a, a, a)
        @test_throws ErrorException optimal_alignment(A, p, p)
        @test_throws ErrorException optimal_alignment!(A, a, p, p)
    end
end
