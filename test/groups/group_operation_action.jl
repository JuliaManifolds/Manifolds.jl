
include("../utils.jl")
include("group_utils.jl")

using Manifolds:
    LeftForwardAction, LeftBackwardAction, RightForwardAction, RightBackwardAction

@testset "Group operation action" begin
    G = GroupManifold(NotImplementedManifold(), Manifolds.MultiplicationOperation())
    A_left_fwd = GroupOperationAction(G)
    A_left_back = GroupOperationAction(G, LeftBackwardAction())
    A_right_fwd = GroupOperationAction(G, RightForwardAction())
    A_right_back = GroupOperationAction(G, RightBackwardAction())

    types = [Matrix{Float64}]

    @test group_manifold(A_left_fwd) === G
    @test base_group(A_left_fwd) == G
    @test repr(A_left_fwd) == "GroupOperationAction($(repr(G)), (LeftAction(), LeftSide()))"
    @test repr(A_right_back) ==
          "GroupOperationAction($(repr(G)), (RightAction(), RightSide()))"

    for type in types
        a_pts = convert.(type, [reshape(i:(i + 3), 2, 2) for i in 1:3])
        m_pts = convert.(type, [reshape((i + 2):(i + 5), 2, 2) for i in 1:3])

        atol = eltype(m_pts[1]) == Float32 ? 1e-5 : 1e-10

        test_action(
            A_left_fwd,
            a_pts,
            m_pts;
            test_optimal_alignment=false,
            test_diff=false,
            atol=atol,
            test_switch_direction=true,
        )

        test_action(
            A_right_back,
            a_pts,
            m_pts;
            test_optimal_alignment=false,
            test_diff=false,
            atol=atol,
            test_switch_direction=true,
        )
    end

    G = SpecialOrthogonal(3)
    M = Rotations(3)
    A_left_fwd = GroupOperationAction(G)
    A_left_back = GroupOperationAction(G, LeftBackwardAction())
    A_right_fwd = GroupOperationAction(G, RightForwardAction())
    A_right_back = GroupOperationAction(G, RightBackwardAction())

    p = Matrix{Float64}(I, 3, 3)
    aω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
    mω = [2 .* ω for ω in aω]
    a_pts = [exp(M, p, hat(M, p, ωi)) for ωi in aω]
    m_pts = [exp(M, p, hat(M, p, ωi)) for ωi in mω]
    X_pts = [
        hat(M, p, [-1.0, 2.0, 0.5]),
        hat(M, p, [2.0, 1.0, 0.5]),
        hat(M, p, [0.5, 0.5, 0.5]),
    ]

    @test group_manifold(A_left_fwd) === G
    @test base_group(A_left_fwd) == G
    @test repr(A_left_fwd) == "GroupOperationAction($(repr(G)), (LeftAction(), LeftSide()))"
    @test repr(A_right_back) ==
          "GroupOperationAction($(repr(G)), (RightAction(), RightSide()))"

    test_action(
        A_left_fwd,
        a_pts,
        m_pts,
        X_pts;
        test_optimal_alignment=true,
        test_diff=true,
        test_switch_direction=true,
    )

    test_action(
        A_left_back,
        a_pts,
        m_pts,
        X_pts;
        test_optimal_alignment=true,
        test_diff=true,
        test_switch_direction=true,
    )

    test_action(
        A_right_fwd,
        a_pts,
        m_pts,
        X_pts;
        test_optimal_alignment=true,
        test_diff=true,
        test_switch_direction=true,
    )

    test_action(
        A_right_back,
        a_pts,
        m_pts,
        X_pts;
        test_optimal_alignment=true,
        test_diff=true,
        test_switch_direction=true,
    )

    m = m_pts[1]
    X = X_pts[1]
    e = identity_element(G)

    @testset "apply_diff" begin
        @test isapprox(M, m, apply_diff(A_left_fwd, e, m, X), X)
        @test isapprox(M, m, inverse_apply_diff(A_left_fwd, e, m, X), X)
        @test isapprox(M, m, apply_diff(A_right_back, e, m, X), X)
        @test isapprox(M, m, inverse_apply_diff(A_right_back, e, m, X), X)

        @test isapprox(M, m, apply_diff(A_left_back, e, m, X), X)
        @test isapprox(M, m, inverse_apply_diff(A_left_back, e, m, X), X)
        @test isapprox(M, m, apply_diff(A_right_fwd, e, m, X), X)
        @test isapprox(M, m, inverse_apply_diff(A_right_fwd, e, m, X), X)

        eX = allocate(X)
        @test apply_diff!(A_left_fwd, eX, e, m, X) === eX
        @test isapprox(M, m, eX, X)
        eX = allocate(X)
        @test inverse_apply_diff!(A_left_fwd, eX, e, m, X) === eX
        @test isapprox(M, m, eX, X)
    end

    @testset "apply_diff_group" begin
        @test apply_diff_group(A_left_fwd, a_pts[1], X, m) ≈
              translate_diff(G, m, a_pts[1], X, RightBackwardAction())
        Y = similar(X)
        apply_diff_group!(A_left_fwd, Y, a_pts[1], X, m)
        @test Y ≈ translate_diff(G, m, a_pts[1], X, RightBackwardAction())

        @test adjoint_apply_diff_group(A_left_fwd, a_pts[1], X, m) ≈
              inverse_translate_diff(G, a_pts[1], m, X, RightBackwardAction())
        Y = similar(X)
        adjoint_apply_diff_group!(A_left_fwd, Y, a_pts[1], X, m)
        @test Y ≈ inverse_translate_diff(G, a_pts[1], m, X, RightBackwardAction())

        @test adjoint_apply_diff_group(A_right_fwd, a_pts[1], X, m) ≈
              inverse_translate_diff(
            G,
            m,
            a_pts[1],
            inv_diff(G, a_pts[1], X),
            RightBackwardAction(),
        )

        Y = similar(X)
        adjoint_apply_diff_group!(A_right_fwd, Y, a_pts[1], X, m)
        @test Y ≈ inverse_translate_diff(G, m, a_pts[1], -X, RightBackwardAction())

        @test apply_diff_group(A_right_fwd, a_pts[1], X, m) ≈ -m \ X * m
        @test apply_diff_group(A_left_back, a_pts[1], X, m) ≈ -X
    end
end
