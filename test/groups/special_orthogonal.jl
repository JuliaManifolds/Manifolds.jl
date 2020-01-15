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
    vpts = [hat(M, x, [-1.0, 2.0, 0.5])]

    for T in types
        gpts = convert.(T, pts)
        vgpts = convert.(T, vpts)
        @test compose(G, gpts[1], gpts[2]) ≈ gpts[1] * gpts[2]
        @test translate_diff(G, gpts[2], gpts[1], vgpts[1], LeftAction()) ≈ vgpts[1]
        @test translate_diff(G, gpts[2], gpts[1], vgpts[1], RightAction()) ≈ transpose(gpts[2]) * vgpts[1] * gpts[2]
        test_group(G, gpts, vgpts; test_diff = true)
    end

    @testset "Decorator forwards to group" begin
        M = NotImplementedGroupDecorator(G)
        @test Manifolds.is_decorator_group(M) === Val(true)
        @test base_group(M) === G
        @test Identity(M) === Identity(G)
        test_group(G, pts, vpts; test_diff = true)
    end

    @testset "Group forwards to decorated" begin
        retraction_methods = [Manifolds.PolarRetraction(),
                              Manifolds.QRRetraction()]

        inverse_retraction_methods = [Manifolds.PolarInverseRetraction(),
                                      Manifolds.QRInverseRetraction()]

        for T in types
            gpts = convert.(T, pts)
            test_manifold(G, gpts;
                test_reverse_diff = false,
                test_injectivity_radius = false,
                test_project_tangent = true,
                test_musical_isomorphisms = false,
                retraction_methods = retraction_methods,
                inverse_retraction_methods = inverse_retraction_methods,
                exp_log_atol_multiplier = 20,
                retraction_atol_multiplier = 12,
            )
        end
    end
end
