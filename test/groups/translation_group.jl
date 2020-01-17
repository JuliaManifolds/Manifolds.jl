include("../utils.jl")
include("group_utils.jl")

@testset "Translation group" begin
    @testset "real" begin
        G = TranslationGroup(2, 3)
        @test repr(G) == "TranslationGroup(2, 3; field = ℝ)"
        @test repr(TranslationGroup(2, 3; field = ℂ)) == "TranslationGroup(2, 3; field = ℂ)"

        types = [Matrix{Float64}]
        @test base_manifold(G) === Euclidean(2, 3)

        pts = [reshape(i:i+5, (2, 3)) for i in 1:3]
        vpts = [reshape(-2:3, (2, 3))]
        for T in types
            gpts = convert.(T, pts)
            vgpts = convert.(T, vpts)
            @test compose(G, gpts[1], gpts[2]) ≈ gpts[1] + gpts[2]
            test_group(G, gpts, vgpts; test_diff = true)
        end
    end

    @testset "complex" begin
        G = TranslationGroup(2, 3; field = ℂ)
        @test repr(G) == "TranslationGroup(2, 3; field = ℂ)"

        types = [Matrix{ComplexF64}]
        @test base_manifold(G) === Euclidean(2, 3; field = ℂ)

        pts = [reshape(complex.(i:i+5, i+1:i+6), (2, 3)) for i in 1:3]
        vpts = [reshape(complex.(-2:3, -1:4), (2, 3))]
        for T in types
            gpts = convert.(T, pts)
            vgpts = convert.(T, vpts)
            @test compose(G, gpts[1], gpts[2]) ≈ gpts[1] + gpts[2]
            @test translate_diff(G, gpts[2], gpts[1], vgpts[1]) ≈ vgpts[1]
            test_group(G, gpts, vgpts; test_diff = true)
        end
    end
end
