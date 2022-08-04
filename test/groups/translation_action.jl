
include("../utils.jl")
include("group_utils.jl")

@testset "Translation action" begin
    M = Euclidean(2, 3)
    G = TranslationGroup(2, 3)
    A = TranslationAction(Euclidean(2, 3), G)

    @test repr(A) == "TranslationAction($(repr(M)), $(repr(G)), LeftAction())"
    @test repr(switch_direction(A)) ==
          "TranslationAction($(repr(M)), $(repr(G)), RightAction())"

    types_a = [Matrix{Float64}]

    types_m = [Matrix{Float64}]

    @test group_manifold(A) == M
    @test base_group(A) == G
    @test base_manifold(G) == M

    for (T_A, T_M) in zip(types_a, types_m)
        a_pts =
            convert.(
                T_A,
                [
                    [0.0 1.0 2.0; 2.0 4.0 1.0],
                    [-1.0 0.0 2.0; -2.0 3.0 3.0],
                    [1.0 1.0 2.0; 3.0 2.0 1.0],
                ],
            )
        a_X_pts =
            convert.(
                T_A,
                [
                    [10.0 1.0 2.0; 2.0 4.0 1.0],
                    [-1.0 10.0 2.0; -2.0 3.0 3.0],
                    [1.0 1.0 12.0; 3.0 2.0 1.0],
                ],
            )
        m_pts =
            convert.(
                T_M,
                [
                    [0.0 1.0 2.0; 2.0 4.0 1.0],
                    [-1.0 0.0 2.0; -2.0 3.0 3.0],
                    [1.0 1.0 2.0; 3.0 2.0 1.0],
                ],
            )
        X_pts =
            convert.(
                T_M,
                [
                    [0.0 1.0 2.0; 2.0 4.0 1.0],
                    [-1.0 0.0 2.0; -2.0 3.0 3.0],
                    [1.0 1.0 2.0; 3.0 2.0 1.0],
                ],
            )

        atol = if eltype(T_M) == Float32
            1e-7
        else
            1e-15
        end
        @test apply_diff(A, a_pts[1], m_pts[1], X_pts[1]) === X_pts[1]
        @test inverse_apply_diff(A, a_pts[1], m_pts[1], X_pts[1]) === X_pts[1]

        @test apply_diff_group(A, a_pts[1], a_X_pts[1], m_pts[1]) === a_X_pts[1]
        @test apply_diff_group(A, Identity(G), a_X_pts[1], m_pts[1]) === a_X_pts[1]
        Y = similar(a_X_pts[1])
        apply_diff_group!(A, Y, a_pts[1], a_X_pts[1], m_pts[1])
        @test Y == a_X_pts[1]
        Y = similar(a_X_pts[1])
        apply_diff_group!(A, Y, Identity(G), a_X_pts[1], m_pts[1])
        @test Y == a_X_pts[1]

        @test adjoint_apply_diff_group(A, a_pts[1], X_pts[1], m_pts[1]) === X_pts[1]
        @test adjoint_apply_diff_group(A, Identity(G), X_pts[1], m_pts[1]) === X_pts[1]
        Y = similar(X_pts[1])
        adjoint_apply_diff_group!(A, Y, a_pts[1], X_pts[1], m_pts[1])
        @test Y == X_pts[1]
        Y = similar(X_pts[1])
        adjoint_apply_diff_group!(A, Y, Identity(G), X_pts[1], m_pts[1])
        @test Y == X_pts[1]

        test_action(
            A,
            a_pts,
            m_pts,
            X_pts;
            test_optimal_alignment=false,
            test_diff=true,
            atol=atol,
        )
    end
end
