using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "The generalized Stiefel manifold" begin
    @test_throws DomainError GeneralizedStiefel(2, 3)

    @testset "Real" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedStiefel(3, 2, B)
        p1 = [1.0 0.0; 0.0 0.5; 0.0 0.0]
        X = [0.0 0.0; 0.0 0.0; 1.0 1.0]
        X2 = [0.0 0.0; 0.0 0.0; -1.0 1.0]
        p2 = retract(M, p1, X)
        p3 = retract(M, p1, X2)
        W = [0.1 0.2; -0.3 0.4; 0.5 -0.6]
        # Invalids
        pf1 = 2 * p1
        pf2 = [1.0, 0.0, 0.0, 0.0]
        pf3 = 1im * [1.0 0.0; 0.0 1.0; 0.0 0.0]
        Xf1 = [1.0 1.0; 0.0 0.0; 0.0 0.0] # p1' * B * Xf1 is not skew-symmetric
        Xf2 = [0.0, 0.0, 1.0, 0.0]
        Xf3 = 1im * zero_vector(M, p1)

        Manifolds.Test.test_manifold(
            M,
            Dict(
                :EmbeddedPoints => [[1.0 0.0; 0.0 2.0; 0.0 0.0]],
                :EmbeddedVectors => [W],
                :Functions => [
                    copy, copyto!, default_retraction_method, embed, get_embedding, inner,
                    inverse_retract, is_flat, is_point, is_vector, manifold_dimension, norm,
                    project, rand, repr, representation_size, retract, zero_vector,
                ],
                :InvalidPoints => AbstractArray[pf1, pf2, pf3], # AbstractArray avoids promotion to complex
                :InvalidVectors => AbstractArray[Xf1, Xf2, Xf3],
                :InverseRetractionMethods => [
                    missing, PolarInverseRetraction(), missing, QRInverseRetraction(),
                ],
                :Points => [p1, p2, p3],
                :RetractionMethods => [
                    CayleyRetraction(), PolarRetraction(), ProjectionRetraction(),
                    QRRetraction(),
                ],
                :SecondVector => X2,
                :Vectors => [X, project(M, p2, X2), project(M, p3, X)],
                :Seed => 42,
                :TestRandScaling => true,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-13,
                default_retraction_method => ProjectionRetraction(),
                (embed, :Point) => p1,
                (embed, :Vector) => X,
                get_embedding => Euclidean(3, 2),
                inner => 0.0,
                is_flat => false,
                manifold_dimension => 3,
                norm => sqrt(2),
                (project, :Point) => p1,
                (project, :Vector) => W - p1 * (p1' * B * W + W' * B * p1) / 2,
                (retract, ProjectionRetraction()) => project(M, p1 + X),
                repr => "GeneralizedStiefel(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℝ)",
                representation_size => (3, 2),
                :IsPointErrors => [DomainError, DomainError, ManifoldDomainError],
                :IsVectorErrors => [DomainError, DomainError, ManifoldDomainError],
                :IsVectorBasepointError => ManifoldDomainError,
            ),
        )

        @testset "Basics" begin
            @test M == GeneralizedStiefel(3, 2, copy(B))
            @test hash(M) == hash(GeneralizedStiefel(3, 2, copy(B)))
            @test M != GeneralizedStiefel(3, 2, B, ℂ)
            Bs = [1.0 -0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
            @test !isequal(M, GeneralizedStiefel(3, 2, Bs))
            @test base_manifold(M) === M
            @test !is_vector(M, p1, p2)
            @test_throws DomainError is_vector(M, p1, p2; error = :error)
            # vector_transport_to is not part of the suite, since it requires an inverse
            # retraction compatible with the default retraction
            @test vector_transport_to(M, p1, X, p2, ProjectionTransport()) == project(M, p2, X)
            M1 = GeneralizedStiefel(3, 1, B)
            q1 = project(M1, reshape([1.0, 1.0, 1.0], 3, 1))
            @test is_vector(M1, q1, project(M1, q1, reshape([0.3, -0.2, 0.5], 3, 1)))
        end

        @testset "Polar retraction" begin
            # the polar factor q of p + X with respect to B: q'B(p + X) is symmetric positive definite
            q = retract(M, p1, X, PolarRetraction())
            @test q' * B * (p1 + X) ≈ (q' * B * (p1 + X))'
            @test isposdef(Symmetric(q' * B * (p1 + X)))
        end

        @testset "QR and Cayley retractions for a diagonal B" begin
            M2 = GeneralizedStiefel(4, 2, Diagonal([1.0, 2.0, 3.0, 4.0]))
            p = project(M2, [1.0 0.0; 1.0 1.0; 1.0 2.0; 0.0 3.0])
            Y = project(M2, p, [0.1 0.2; -0.3 0.4; 0.5 0.6; 0.7 -0.8])
            q = retract(M2, p, Y, PolarRetraction())
            @test is_point(M2, q)
            @test isapprox(M2, p, inverse_retract(M2, p, q, PolarInverseRetraction()), Y)
            r = retract(M2, p, Y, QRRetraction())
            @test is_point(M2, r)
            # r = (p + Y)R^{-1}, so r'B(p + Y) = R is upper triangular with positive diagonal
            R = r' * M2.B * (p + Y)
            @test isapprox(tril(R, -1), zeros(2, 2); atol = 1.0e-14)
            @test all(diag(R) .> 0)
            @test isapprox(M2, p, inverse_retract(M2, p, r, QRInverseRetraction()), Y)
            c = retract(M2, p, Y, CayleyRetraction())
            @test is_point(M2, c)
            @test retract(M2, p, zero(Y), CayleyRetraction()) == p
            @test retract(M2, p, Y / 2, CayleyRetraction()) ≈
                ManifoldsBase.retract_fused(M2, p, Y, 0.5, CayleyRetraction())
        end
    end

    @testset "Complex" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedStiefel(3, 2, B, ℂ)
        p1 = [1im 0.0; 0.0 0.5im; 0.0 0.0]
        X = project(M, p1, [0.1 0.2im; -0.3 0.4; 0.5im 0.6])
        p2 = retract(M, p1, X, PolarRetraction())
        p3 = retract(M, p1, project(M, p1, [0.0 0.0; 0.0 0.0; 0.3 -0.2im]))
        F = ComplexF64[1.0 2.0im; 3.0 4.0; 5.0im 6.0]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :EmbeddedPoints => [F],
                :Functions => [
                    copy, copyto!, embed, get_embedding, inner, is_flat, is_point, is_vector,
                    manifold_dimension, norm, project, rand, repr, representation_size,
                    retract, zero_vector,
                ],
                :InvalidPoints => AbstractArray[2 * p1, [1.0, 0.0, 0.0, 0.0]],
                :InvalidVectors => AbstractArray[
                    [0.0 1.0; 1.0 0.0; 0.0 0.0], 1.0e-9 * [0.0 1.0; 1.0 0.0; 0.0 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                :Points => [p1, p2, p3],
                :RetractionMethods => [PolarRetraction(), ProjectionRetraction()],
                :Vectors => [X, project(M, p2, X), project(M, p3, X)],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-13,
                get_embedding => Euclidean(3, 2; field = ℂ),
                is_flat => false,
                manifold_dimension => 8,
                (number_eltype, rand) => ComplexF64,
                repr => "GeneralizedStiefel(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℂ)",
                representation_size => (3, 2),
                :IsPointErrors => [DomainError, DomainError],
                :IsVectorErrors => [DomainError, DomainError, DomainError],
            ),
        )

        @testset "project with the identity as B" begin
            MI = GeneralizedStiefel(3, 2, Matrix{ComplexF64}(I, 3, 3), ℂ)
            @test is_point(MI, project(MI, F))
        end
    end

    @testset "Quaternion" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedStiefel(3, 2, B, ℍ)
        Manifolds.Test.test_manifold(
            M,
            Dict(:Functions => [is_flat, manifold_dimension, repr, representation_size]),
            # Expectations
            Dict(
                is_flat => false,
                manifold_dimension => 18,
                repr => "GeneralizedStiefel(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℍ)",
                representation_size => (3, 2),
            ),
        )
    end

    @testset "field parameter" begin
        B = [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0]
        M = GeneralizedStiefel(3, 2, B; parameter = :field)
        @test repr(M) ==
            "GeneralizedStiefel(3, 2, [1.0 0.0 0.0; 0.0 4.0 0.0; 0.0 0.0 1.0], ℝ; parameter=:field)"
        @test typeof(get_embedding(M)) === Euclidean{ℝ, Tuple{Int64, Int64}}
    end
end
