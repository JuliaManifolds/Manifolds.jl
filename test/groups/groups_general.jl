using StaticArrays: identity_perm
using Base: decode_overlong

include("../header.jl")
include("group_utils.jl")

using Manifolds:
    LeftForwardAction, LeftBackwardAction, RightForwardAction, RightBackwardAction

@testset "General group tests" begin
    @testset "Not implemented operation" begin
        G = GroupManifold(
            NotImplementedManifold(),
            NotImplementedOperation(),
            Manifolds.LeftInvariantRepresentation(),
        )
        @test repr(G) ==
              "GroupManifold(NotImplementedManifold(), NotImplementedOperation())"
        p = [1.0, 2.0]
        X = [2.0, 3.0]
        eg = Identity(G)
        @test repr(eg) === "Identity(NotImplementedOperation)"
        @test adjoint(eg) == eg
        @test number_eltype(eg) == Bool
        @test !is_group_manifold(NotImplementedManifold())
        @test !is_group_manifold(NotImplementedManifold(), NotImplementedOperation())
        @test !has_biinvariant_metric(NotImplementedManifold())
        @test !has_invariant_metric(NotImplementedManifold(), LeftForwardAction())
        @test is_identity(G, eg) # identity transparent
        @test_throws ErrorException identity_element(G) # but for a NotImplOp there is no concrete id.
        @test isapprox(G, eg, eg)
        @test !isapprox(G, Identity(AdditionOperation()), eg)
        @test !isapprox(G, Identity(AdditionOperation()), eg)
        @test !isapprox(
            G,
            Identity(AdditionOperation()),
            Identity(MultiplicationOperation()),
        )
        @test_throws DomainError is_point(G, Identity(AdditionOperation()); error=:error)
        @test is_point(G, eg)
        @test_throws ErrorException is_identity(G, 1) # same error as before i.e. dispatch isapprox works
        @test Manifolds.check_size(G, eg) === nothing
        @test Manifolds.check_size(
            Manifolds.EmptyTrait(),
            MetricManifold(NotImplementedManifold(), EuclideanMetric()),
            eg,
        ) isa DomainError
        @test !is_vector(G, Identity(AdditionOperation()), X)
        # wrong identity
        @test_throws DomainError is_vector(
            G,
            Identity(AdditionOperation()),
            X;
            error=:error,
        )
        # identity_element for G not implemented
        @test_throws ErrorException is_vector(G, eg, X; error=:error)
        @test Identity(NotImplementedOperation()) === eg
        @test Identity(NotImplementedOperation) === eg
        @test !is_point(G, Identity(AdditionOperation()))
        @test !isapprox(G, eg, Identity(AdditionOperation()))
        @test !isapprox(G, Identity(AdditionOperation()), eg)

        @test NotImplementedOperation(
            NotImplementedManifold(),
            Manifolds.LeftInvariantRepresentation(),
        ) === G
        @test (NotImplementedOperation())(
            NotImplementedManifold(),
            Manifolds.LeftInvariantRepresentation(),
        ) === G

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

        @test_throws MethodError inv!(G, p, p)
        @test_throws MethodError inv!(G, p, eg)
        @test_throws MethodError inv(G, p)
        @test_throws MethodError inv_diff(G, p, X)
        @test_throws MethodError inv_diff!(G, X, p, X)

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
        @test_throws MethodError translate(G, p, p, LeftForwardAction())
        @test_throws MethodError translate(G, p, p, RightBackwardAction())
        @test_throws MethodError translate!(G, p, p, p)
        @test_throws MethodError translate!(G, p, p, p, LeftForwardAction())
        @test_throws MethodError translate!(G, p, p, p, RightBackwardAction())

        @test_throws MethodError inverse_translate(G, p, p)
        @test_throws MethodError inverse_translate(G, p, p, LeftForwardAction())
        @test_throws MethodError inverse_translate(G, p, p, RightBackwardAction())
        @test_throws MethodError inverse_translate!(G, p, p, p)
        @test_throws MethodError inverse_translate!(G, p, p, p, LeftForwardAction())
        @test_throws MethodError inverse_translate!(G, p, p, p, RightBackwardAction())

        @test_throws MethodError adjoint_action(G, p, X)
        @test_throws MethodError adjoint_action(G, p, X, LeftAction())
        @test_throws MethodError adjoint_action(G, p, X, RightAction())
        @test_throws MethodError adjoint_action!(G, p, X, X)
        @test_throws MethodError adjoint_action!(G, p, X, X, LeftAction())
        @test_throws MethodError adjoint_action!(G, p, X, X, RightAction())

        @test_throws MethodError exp_lie(G, X)
        @test_throws MethodError exp_lie!(G, p, X)
        # no transparency error, but _log_lie missing
        @test_throws MethodError log_lie(G, p)
        @test_throws MethodError log_lie!(G, X, p)
    end

    @testset "Action direction" begin
        @test switch_direction(LeftAction()) === RightAction()
        @test switch_direction(RightAction()) === LeftAction()

        G = GroupManifold(NotImplementedManifold(), NotImplementedOperation())
        @test Manifolds._action_order(G, 1, 2, LeftForwardAction()) === (1, 2)
        @test Manifolds._action_order(G, 1, 2, RightBackwardAction()) === (2, 1)
    end

    @testset "Action side" begin
        @test switch_side(LeftSide()) === RightSide()
        @test switch_side(RightSide()) === LeftSide()
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
        @test identity_element(G, 2.0) == 0.0
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
        G = GroupManifold(
            NotImplementedManifold(),
            Manifolds.MultiplicationOperation(),
            Manifolds.LeftInvariantRepresentation(),
        )
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
        @test inv(ge) === ge
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

    @testset "Issue #669" begin
        G = SpecialOrthogonal(3)

        id = identity_element(G)
        X = hat(G, Identity(G), [1.0, 2.0, 3.0])
        function apply_at_id(X, d, s)
            A = GroupOperationAction(G, (d, s))
            return apply_diff_group(A, id, X, id)
        end
        function apply_at_id!(Y, X, d, s)
            A = GroupOperationAction(G, (d, s))
            return apply_diff_group!(A, Y, id, X, id)
        end

        _get_sign(::LeftAction, ::LeftSide) = 1 # former case
        _get_sign(::LeftAction, ::RightSide) = -1 # new case
        _get_sign(::RightAction, ::LeftSide) = -1 # new case
        _get_sign(::RightAction, ::RightSide) = 1 # former case
        Y = similar(X)

        for d in [LeftAction(), RightAction()]
            for s in [LeftSide(), RightSide()]
                @test apply_at_id(X, d, s) ≈ _get_sign(d, s) * X
                apply_at_id!(Y, X, d, s)
                @test Y ≈ _get_sign(d, s) * X
            end
        end
    end

    @testset "Jacobians" begin
        M = SpecialOrthogonal(3)
        p = [
            -0.333167290022488 -0.7611396995437196 -0.5564763378954822
            0.8255218425902797 0.049666662385985494 -0.5621804959741897
            0.4555362161951786 -0.646683524164589 0.6117901399243428
        ]
        # testing the fallback definition just in case
        @test invoke(adjoint_matrix, Tuple{AbstractManifold,Any}, M, p) ≈ p
    end
end

struct NotImplementedAction <: AbstractGroupAction{LeftAction} end

@testset "General group action tests" begin
    @testset "Not implemented operations" begin
        A = NotImplementedAction()
        p = [1.0, 2.0]
        a = [1.0, 2.0]
        X = [1.0, 2.0]

        @test_throws MethodError apply(A, a, p)
        @test_throws MethodError apply!(A, p, a, p)
        @test_throws MethodError inverse_apply(A, a, p)
        @test_throws MethodError inverse_apply!(A, p, a, p)
        @test_throws MethodError apply_diff(A, a, p, X)
        @test_throws MethodError apply_diff!(A, X, p, a, X)
        @test_throws MethodError inverse_apply_diff(A, a, p, X)
        @test_throws MethodError inverse_apply_diff!(A, X, p, a, X)
        @test_throws MethodError compose(A, a, a)
        @test_throws MethodError compose!(A, a, a, a)
        @test_throws MethodError optimal_alignment(A, p, p)
        @test_throws MethodError optimal_alignment!(A, a, p, p)
    end
end
