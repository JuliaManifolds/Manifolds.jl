include("utils.jl")

@testset "Symmetric Positive Definite" begin
    M = Manifolds.SymmetricPositiveDefinite(3)

    types = [   Matrix{Float64},
                Matrix{Float32},
            ]
    for T in types
        A(α) = [1. 0. 0.; 0. cos(α) sin(α); 0. -sin(α) cos(α)]
        ptsF = [#
            [1. 0. 0.; 0. 1. 0.; 0. 0. 1],
            [2. 0. 0.; 0. 2. 0.; 0. 0. 1],
            A(π/6) * [1. 0. 0.; 0. 2. 0.; 0. 0. 1] * transpose(A(π/6)),
            ]
        pts = [convert(T, a) for a in ptsF]
        test_manifold(M, pts;
                test_vector_transport = true,
                test_forward_diff = false,
                test_reverse_diff = false,
                exp_log_atol_multiplier = 8
        )
    end
end
