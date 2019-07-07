include("utils.jl")

@testset "Symmetric Positive Definite" begin
    M = Manifolds.SymmetricPositiveDefinite(3)

    types = [   Matrix{Float64},
#                SizedMatrix{3, 3, Float64},
#                MMatrix{3, 3, Float64},
#                Matrix{Float32},
#                SizedMatrix{3, 3, Float32},
#                MMatrix{3, 3, Float32}
            ]
    for T in types
        A(α) = [1. 0. 0.; 0. cos(α) sin(α); 0. -sin(α) cos(α)]
        ptsF = [#
            [1. 0. 0.; 0. 1. 0.; 0. 0. 1],
            [2. 0. 0.; 0. 2. 0.; 0. 0. 1],
            A(π/6) * [1. 0. 0.; 0. 2. 0.; 0. 0. 1] * transpose(A(π/6)),
            ]
        pts = [convert(T, a) for a in ptsF]
        test_manifold(M, pts)
    end
end
