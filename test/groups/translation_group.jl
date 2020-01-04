include("../utils.jl")
include("group_utils.jl")

@testset "Translation group" begin
    @testset "real" begin
        G = TranslationGroup(2, 3)
        @test repr(G) == "TranslationGroup(2, 3; field = ℝ)"

        types = [Matrix{Float64}]
        @test base_manifold(G) === Euclidean(2, 3)

        pts = [reshape(i:i+5, (2, 3)) for i in 1:3]
        for T in types
            gpts = convert.(T, pts)
            @test compose(G, gpts[1], gpts[2]) ≈ gpts[1] + gpts[2]
            test_group(G, gpts)
        end
    end

    @testset "complex" begin
        G = TranslationGroup(2, 3; field = ℂ)
        @test repr(G) == "TranslationGroup(2, 3; field = ℂ)"

        types = [Matrix{ComplexF64}]
        @test base_manifold(G) === Euclidean(2, 3; field = ℂ)

        pts = [reshape(complex.(i:i+5, i+1:i+6), (2, 3)) for i in 1:3]
        for T in types
            gpts = convert.(T, pts)
            @test compose(G, gpts[1], gpts[2]) ≈ gpts[1] + gpts[2]
            test_group(G, gpts)
        end
    end
end
