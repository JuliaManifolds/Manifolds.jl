using LinearAlgebra, Manifolds, Random, Test

@testset "Veronese Manifold" begin
    @test Veronese(4, 3) == Veronese(4, 3; parameter = :type)
    @test typeof(Veronese(4, 3)) !== typeof(Veronese(5, 3))
    @test typeof(Veronese(4, 3; parameter = :field)) ===
        typeof(Veronese(5, 3; parameter = :field))
    @test_throws ArgumentError Veronese(0, 2)
    @test_throws ArgumentError Veronese(3, 0)
    @test_throws ArgumentError Veronese(3, 2; parameter = :invalid)

    n = 4

    x = normalize([1.0, 2.0, -1.0, 0.5])
    u = normalize([x[2], -x[1], 0.0, 0.0])
    v_raw = [0.5, -1.0, 2.0, 1.0]
    v = v_raw - dot(x, v_raw) * x

    z = normalize([1.0, -0.5, 0.25, 2.0])
    w_raw = [-0.5, 1.0, 0.5, 0.25]
    w = w_raw - dot(z, w_raw) * z

    for parameter in (:type, :field), d in 1:3
        M = Veronese(n, d; parameter)

        p = [[-0.7], x]
        q = [[0.9], z]
        X = [[0.2], u]
        Y = [[-0.3], v]
        Z = [[0.1], w]

        invalid_points = Any[
            [[1.0, 2.0], x],
            [[0.0], x],
            [[Inf], x],
            [[1.0], x[1:3]],
            [[1.0], 2 .* x],
        ]
        invalid_vectors = Any[
            [[0.2, 0.1], u],
            [[0.2], x],
        ]

        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Aliased => false,
                :Functions => [
                    embed,
                    get_embedding,
                    inner,
                    is_point,
                    is_vector,
                    manifold_dimension,
                    rand,
                    repr,
                ],
                :InvalidPoints => invalid_points,
                :InvalidVectors => invalid_vectors,
                :Points => [p, q],
                :Rng => Random.Xoshiro(42),
                :SecondVector => Y,
                :Vectors => [X, Z],
            ),
            Dict(
                :atol => 1.0e-12,
                :IsPointErrors => fill(DomainError, length(invalid_points)),
                :IsVectorErrors => fill(DomainError, length(invalid_vectors)),
                get_embedding => Euclidean(n^d; parameter),
                manifold_dimension => n,
                repr => "Veronese($n, $d$(parameter === :field ? "; parameter=:field" : ""))",
            ),
        )

        @test Manifolds.get_parameter_type(M) === parameter

        @test_throws DomainError is_vector(M, invalid_points[1], X; error = :error)

        @testset "Equivalent representatives" begin
            p_embedded = embed(M, p)

            p_equivalent = [[(-1)^d * p[1][1]], -x]
            @test embed(M, p_equivalent) ≈ p_embedded

            p_flipped = [[p[1][1]], -x]
            @test embed(M, p_flipped) ≈ (-1)^d .* p_embedded
        end

        @testset "Induced metric" begin
            X_embedded = embed(M, p, X)
            Y_embedded = embed(M, p, Y)

            @test inner(M, p, X, Y) ≈ dot(X_embedded, Y_embedded)
        end

        @testset "Embedding differential" begin
            p_embedded = embed(M, p)
            X_embedded = embed(M, p, X)

            t = 1.0e-7
            x_t = exp(Sphere(n - 1), x, t .* u)
            p_t = [[p[1][1] + t * X[1][1]], x_t]
            embedding_difference = (embed(M, p_t) - p_embedded) ./ t

            @test embedding_difference ≈ X_embedded rtol = 1.0e-6 atol = 1.0e-7
        end
    end

    @testset "Float32 rand" begin
        for parameter in (:type, :field)
            M = Veronese(3, 2; parameter)
            p = [Float32[-0.5], normalize(Float32[1, 2, 3])]
            X = rand(Random.Xoshiro(12), M; vector_at = p)

            @test all(eltype(Xᵢ) === Float32 for Xᵢ in X)
        end
    end
end
