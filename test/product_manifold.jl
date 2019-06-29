include("utils.jl")

@testset "Product manifold" begin
    M1 = Manifolds.Sphere(2)
    M2 = Manifolds.Euclidean(2)
    M = Manifolds.ProductManifold(M1, M2)

    types = [Vector{Float64},
             SizedVector{5, Float64},
             MVector{5, Float64},
             Vector{Float32},
             SizedVector{5, Float32},
             MVector{5, Float32}]

    for T in types
        @testset "Type $T" begin
            pts = [convert(T, [1.0, 0.0, 0.0, 0.0, 0.0]),
                   convert(T, [0.0, 1.0, 0.0, 1.0, 0.0]),
                   convert(T, [0.0, 0.0, 1.0, 0.0, 0.1])]
            test_manifold(M,
                          pts,
                          test_reverse_diff = isa(T, Vector))
        end
    end
end
