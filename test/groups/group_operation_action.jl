
include("../utils.jl")
include("group_utils.jl")

@testset "Group operation action" begin
    G = GroupManifold(NotImplementedManifold(), Manifolds.MultiplicationOperation())
    A_left = GroupOperationAction(G)
    A_right = GroupOperationAction(G, RightAction())

    types = [Matrix{Float64}]

    @test group_manifold(A_left) === G
    @test base_group(A_left) == G
    @test repr(A_left) == "GroupOperationAction($(repr(G)), LeftAction())"
    @test repr(A_right) == "GroupOperationAction($(repr(G)), RightAction())"

    @test switch_direction(LeftAction()) == RightAction()
    @test switch_direction(RightAction()) == LeftAction()

    for type in types
        a_pts = convert.(type, [reshape(i:(i + 3), 2, 2) for i in 1:3])
        m_pts = convert.(type, [reshape((i + 2):(i + 5), 2, 2) for i in 1:3])

        atol = eltype(m_pts[1]) == Float32 ? 1e-5 : 1e-10

        test_action(
            A_left,
            a_pts,
            m_pts;
            test_optimal_alignment=false,
            test_diff=false,
            atol=atol,
        )

        test_action(
            A_right,
            a_pts,
            m_pts;
            test_optimal_alignment=false,
            test_diff=false,
            atol=atol,
        )
    end

    G = SpecialOrthogonal(3)
    M = Rotations(3)
    A_left = GroupOperationAction(G)
    A_right = GroupOperationAction(G, RightAction())

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

    @test group_manifold(A_left) === G
    @test base_group(A_left) == G
    @test repr(A_left) == "GroupOperationAction($(repr(G)), LeftAction())"
    @test repr(A_right) == "GroupOperationAction($(repr(G)), RightAction())"

    test_action(A_left, a_pts, m_pts, X_pts; test_optimal_alignment=true, test_diff=true)

    test_action(A_right, a_pts, m_pts, X_pts; test_optimal_alignment=true, test_diff=true)

    @testset "apply_diff_group" begin
        @test apply_diff_group(A_left, a_pts[1], X_pts[1], m_pts[1]) ≈
              translate_diff(G, m_pts[1], a_pts[1], X_pts[1], RightAction())
        Y = similar(X_pts[1])
        apply_diff_group!(A_left, Y, a_pts[1], X_pts[1], m_pts[1])
        @test Y ≈ translate_diff(G, m_pts[1], a_pts[1], X_pts[1], RightAction())

        @test adjoint_apply_diff_group(A_left, a_pts[1], X_pts[1], m_pts[1]) ≈
              inverse_translate_diff(G, a_pts[1], m_pts[1], X_pts[1], RightAction())
        Y = similar(X_pts[1])
        adjoint_apply_diff_group!(A_left, Y, a_pts[1], X_pts[1], m_pts[1])
        @test Y ≈ inverse_translate_diff(G, a_pts[1], m_pts[1], X_pts[1], RightAction())
    end
end
