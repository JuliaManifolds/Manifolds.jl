
include("../utils.jl")
include("group_utils.jl")

@testset "Translation action" begin
    M = Euclidean(2,3)
    G = TranslationGroup(2,3)
    A = TranslationAction(Euclidean(2,3), G)

    @test repr(A) == "TranslationAction($(repr(G)), LeftAction())"
    @test repr(switch_direction(A)) == "TranslationAction($(repr(G)), RightAction())"

    types_a = [
        Matrix{Float64},
        MMatrix{2, 3, Float64},
        Matrix{Float32},
    ]

    types_m = [
        Matrix{Float64},
        MMatrix{2, 3, Float64},
        Matrix{Float32},
    ]

    @test g_manifold(A) == M
    @test base_group(A) == G
    @test base_manifold(G) == M

    for (T_A, T_M) in zip(types_a, types_m)
        a_pts = convert.(T_A, [[0.0 1.0 2.0; 2.0 4.0 1.0], [-1.0 0.0 2.0; -2.0 3.0 3.0], [1.0 1.0 2.0; 3.0 2.0 1.0]])
        m_pts = convert.(T_M, [[0.0 1.0 2.0; 2.0 4.0 1.0], [-1.0 0.0 2.0; -2.0 3.0 3.0], [1.0 1.0 2.0; 3.0 2.0 1.0]])
        v_pts = convert.(T_M, [[0.0 1.0 2.0; 2.0 4.0 1.0], [-1.0 0.0 2.0; -2.0 3.0 3.0], [1.0 1.0 2.0; 3.0 2.0 1.0]])

        atol_inv = if eltype(T_M) == Float32
            1e-7
        else
            1e-15
        end
        test_action(A, a_pts, m_pts, v_pts;
            test_optimal_alignment = false,
            test_diff = true,
            atol_inv = atol_inv)
    end

end
