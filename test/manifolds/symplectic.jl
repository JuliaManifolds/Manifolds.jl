include("../utils.jl")

@testset "Symplectic" begin
    @testset "Real" begin
        small_n = 1
        M = Symplectic(2*small_n)
        M2 = MetricManifold(M, EuclideanMetric())
        p = [0.0 1.0/2.0; -2.0 -2.0]
        @testset "Basics" begin

            @test repr(M) == "Symplectic($(2*small_n), ℝ)"
            @test representation_size(M) == (2small_n, 2small_n)
            @test base_manifold(M) === M

            @test is_point(M, p)
            @test !is_point(M, p + I)
        end
        @testset "Embedding and Projection" begin
            x = [0.0 1.0/2.0; -2.0 -2.0]
            y = similar(x)
            z = embed(M, x)
            @test z == x

            # Make non-trivial Tangent vector.
            X = [-0.121212  0.121212;
                  0.969697 -1.0]
            @test is_vector(M, p, X; atol=1.0e-6)

            Y = similar(X)
            embed!(M, Y, p, X)
            @test Y ≈ X
        end
    end
end
