
include("../utils.jl")
include("group_utils.jl")

@testset "Rotation action" begin
    M = Rotations(2)
    G = SpecialOrthogonal(2)
    A_left = RotationAction(Euclidean(2), G)
    A_right = RotationAction(Euclidean(2), G, RightAction())

    types_a = [Matrix{Float64},
               SizedMatrix{2, 2, Float64},
               MMatrix{2, 2, Float64},
               Matrix{Float32},
               SizedMatrix{2, 2, Float32},
               MMatrix{2, 2, Float32}]

    types_m = [Vector{Float64},
               SizedVector{2, Float64},
               MVector{2, Float64},
               Vector{Float32},
               SizedVector{2, Float32},
               MVector{2, Float32}]


    @test g_manifold(A_left) == Euclidean(2)
    @test base_group(A_left) == G
    @test isa(A_left, AbstractGroupAction{LeftAction})
    @test base_manifold(G) == M

    for (i, T_A, T_M) in zip(1:length(types_a), types_a, types_m)
        angles = (0.0, π/2, 2π/3, π/4)
        a_pts = [convert(T_A, [cos(ϕ) -sin(ϕ); sin(ϕ) cos(ϕ)]) for ϕ in angles]
        m_pts = [convert(T_M, [0.0, 1.0]), convert(T_M, [-1.0, 0.0]), convert(T_M, [1.0, 1.0])]

        atol_inv = if eltype(T_M) == Float32
            1e-7
        else
            1e-15
        end
        test_action(A_left, a_pts, m_pts;
            test_optimal_alignment = true,
            atol_inv = atol_inv)

        if i == 1
            test_action(A_right, a_pts, transpose.(m_pts);
                test_optimal_alignment = false,
                atol_inv = atol_inv)
        end
    end

end
