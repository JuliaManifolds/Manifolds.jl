include("../utils.jl")
include("group_utils.jl")

@testset "Special Orthogonal group" begin
    G = SpecialOrthogonal(3)
    @test repr(G) == "SpecialOrthogonal(3)"
    M = base_manifold(G)
    @test M === Rotations(3)
    x = Matrix(I, 3, 3)

    types = [Matrix{Float64}]
    ω = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]]
    pts = [exp(M, x, hat(M, x, ωi)) for ωi in ω]
    for T in types
        gpts = convert.(T, pts)
        @test compose(G, gpts[1], gpts[2]) ≈ gpts[1] * gpts[2]
        test_group(G, gpts)
    end
end
