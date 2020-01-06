
include("../utils.jl")
include("group_utils.jl")

@testset "Group operation action" begin
    G = GroupManifold(NotImplementedManifold(), Manifolds.MultiplicationOperation())
    A_left = GroupOperationAction(G)
    A_right = GroupOperationAction(G, RightAction())
    @test direction(A_left) === LeftAction()

    types = [Matrix{Float64}, MMatrix{2,2,Float64}, Matrix{Float32}]

    @test g_manifold(A_left) === G
    @test base_group(A_left) == G

    a_pts = []

    for type in types
        a_pts = convert.(type, [reshape(i:i+3, 2, 2) for i = 1:3])
        m_pts = convert.(type, [reshape(i+2:i+5, 2, 2) for i = 1:3])

        atol_inv = eltype(m_pts[1]) == Float32 ? 1e-5 : 1e-10

        test_action(
            A_left,
            a_pts,
            m_pts;
            test_optimal_alignment = false,
            test_diff = false,
            atol_inv = atol_inv,
        )

        test_action(
            A_right,
            a_pts,
            m_pts;
            test_optimal_alignment = false,
            test_diff = false,
            atol_inv = atol_inv,
        )
    end
end
