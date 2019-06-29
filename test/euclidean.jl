include("utils.jl")

@testset "Euclidean" begin
    M = Manifolds.Euclidean(3)
    types = [Vector{Float64},
             SizedVector{3, Float64},
             MVector{3, Float64},
             Vector{Float32},
             SizedVector{3, Float32},
             MVector{3, Float32},
             Vector{Double64},
             MVector{3, Double64},
             SizedVector{3, Double64}]
    for T in types
        @testset "Type $T" begin
            pts = [convert(T, [1.0, 0.0, 0.0]),
                   convert(T, [0.0, 1.0, 0.0]),
                   convert(T, [0.0, 0.0, 1.0])]
            test_manifold(M,
                          pts,
                          test_reverse_diff = isa(T, Vector))
        end
    end

end
