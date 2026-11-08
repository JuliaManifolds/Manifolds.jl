using LinearAlgebra, Manifolds, ManifoldsBase, Random, StaticArrays, Test

@testset "The manifold of Hermitian positive definite matrices" begin
    @test_throws DomainError HermitianPositiveDefinite(-1)
    @test HermitianPositiveDefinite(2) === HermitianPositiveDefinite(2, ℂ)
    @test HermitianPositiveDefinite(2, ℝ) === SymmetricPositiveDefinite(2)
    @test repr(HermitianPositiveDefinite(3; parameter = :field)) ==
        "HermitianPositiveDefinite(3; parameter=:field)"

    M = HermitianPositiveDefinite(3)
    α, φ = π / 6, π / 4
    U = ComplexF64[1 0 0; 0 cos(α) exp(im * φ) * sin(α); 0 -exp(-im * φ) * sin(α) cos(α)]
    p1 = Matrix{ComplexF64}(I, 3, 3)
    p2 = ComplexF64[2 0 0; 0 2 0; 0 0 1]
    p3 = U * ComplexF64[1 0 0; 0 2 0; 0 0 1] * U'
    X = ComplexF64[log(2) 0 0; 0 log(2) 0; 0 0 0] # log(M, p1, p2)
    X2 = ComplexF64[1 1 + im 0.5; 1 - im 1 0; 0.5 0 1]
    Y = ComplexF64[0.5 0.2im 0.2; -0.2im -1 0.3; 0.2 0.3 0.5]
    # Invalids
    pf1 = zeros(ComplexF64, 2, 3) # wrong size
    pf2 = ComplexF64[1 0 0; 0 0 0; 0 0 1] # not positive definite
    pf3 = ComplexF64[2 im 0; im 1 0; 0 0 4] # not Hermitian
    invalid_points = [pf1, pf2, pf3]
    invalid_vectors = [pf1, pf3]
    is_point_errors = [ManifoldDomainError, DomainError, DomainError]
    is_vector_errors = [ManifoldDomainError, DomainError]
    basis = DefaultOrthonormalBasis()
    # coordinates of X in the basis: (1,1), (1,2) real, (1,2) imaginary, (1,3) real, (1,3) imaginary, (2,2), ...
    cX = [log(2), 0.0, 0.0, 0.0, 0.0, log(2), 0.0, 0.0, 0.0]

    @testset "Affine invariant metric with $T points" for T in
        [Matrix{ComplexF64}, MatrixSqrtManifoldPoint]
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Bases => [basis],
                :Coordinates => [cX],
                :Functions => [
                    copy, copyto!, default_inverse_retraction_method,
                    default_retraction_method, default_vector_transport_method, distance,
                    exp, geodesic, get_basis, get_coordinates, get_embedding, get_vector,
                    get_vectors, injectivity_radius, inner, inverse_retract, is_flat,
                    is_point, is_vector, log, manifold_dimension, mid_point, norm,
                    parallel_transport_direction, parallel_transport_to, rand, repr,
                    representation_size, retract, shortest_geodesic,
                    vector_transport_direction, vector_transport_to, zero_vector,
                    # embedding is not yet available for MatrixSqrtManifoldPoint
                    (T === MatrixSqrtManifoldPoint ? () : (embed,))...,
                ],
                :InvalidPoints => invalid_points,
                :InvalidVectors => invalid_vectors,
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :Points => [convert(T, p) for p in [p1, p2, p3]],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => X2,
                :Vectors => [X, Y, X2],
                :VectorTransportMethods => [
                    ParallelTransport(), PoleLadderTransport(), SchildsLadderTransport(),
                ],
                :Seed => 42,
            ),
            # Expectations
            Dict(
                :atol => 1.0e-12,
                default_inverse_retraction_method => LogarithmicInverseRetraction(),
                default_retraction_method => ExponentialRetraction(),
                default_vector_transport_method => ParallelTransport(),
                distance => sqrt(2) * log(2),
                exp => convert(T, p2),
                (get_coordinates, basis) => cX,
                (get_vectors, basis) => :Orthonormal,
                get_embedding => Euclidean(3, 3; field = ℂ),
                injectivity_radius => Inf,
                (injectivity_radius, ExponentialRetraction()) => Inf,
                inner => 2 * log(2),
                is_flat => false,
                log => X,
                manifold_dimension => 9,
                mid_point => convert(T, ComplexF64[sqrt(2) 0 0; 0 sqrt(2) 0; 0 0 1]),
                norm => sqrt(2) * log(2),
                (number_eltype, rand) => ComplexF64,
                # parallel transport from the identity to p2 is X ↦ p2^(1/2) X p2^(1/2)
                parallel_transport_to => 2 * X,
                parallel_transport_direction => 2 * X,
                (vector_transport_to, ParallelTransport()) => 2 * X,
                # the pole ladder is exact on a symmetric space
                (vector_transport_to, PoleLadderTransport()) => 2 * X,
                repr => "HermitianPositiveDefinite(3)",
                representation_size => (3, 3),
                :IsPointErrors => is_point_errors,
                :IsVectorErrors => is_vector_errors,
                :IsVectorBasepointError => ManifoldDomainError,
            ),
        )
    end

    @testset "Affine invariant metric: further properties" begin
        @test sectional_curvature_max(M) == 0.0
        @test manifold_volume(M) == Inf
        @test is_flat(HermitianPositiveDefinite(1))
        @test is_flat(MetricManifold(HermitianPositiveDefinite(1), AffineInvariantMetric()))
        @test !is_flat(MetricManifold(M, AffineInvariantMetric()))
        @testset "Orthonormal basis away from the identity" begin
            B = get_basis(M, p3, basis)
            Ξ = get_vectors(M, p3, B)
            @test length(Ξ) == 9
            @test all(Z -> Z ≈ Z', Ξ)
            G = [inner(M, p3, Ξ[i], Ξ[j]) for i in 1:9, j in 1:9]
            @test G ≈ I atol = 1.0e-12
            c = get_coordinates(M, p3, X2, basis)
            @test c isa Vector{Float64}
            @test get_vector(M, p3, c, basis) ≈ X2
        end
        @testset "Points that are Hermitian up to rounding" begin
            q = p3 + 1.0e-15 * ComplexF64[0 1 0; 0 0 0; 0 0 0]
            @test is_point(M, q)
        end
        @testset "Metric change" begin
            Y = change_metric(M, EuclideanMetric(), p3, X2)
            @test Y ≈ sqrt(Hermitian(p3)) * X2 * sqrt(Hermitian(p3))
            @test is_vector(M, p3, Y)
            @test norm(M, p3, Y) ≈ norm(X2)
            @test change_representer(M, EuclideanMetric(), p3, X2) == p3 * X2 * p3
        end
        @testset "Projection onto the tangent space" begin
            Z = project(M, p1, ComplexF64[1 2 + 2im 1; 0 1 0; 0 0 1])
            @test is_vector(M, p1, Z)
            @test Z ≈ ComplexF64[1 1 + im 0.5; 1 - im 1 0; 0.5 0 1]
        end
        @testset "rand" begin
            q = rand(M)
            @test q isa Matrix{ComplexF64}
            @test is_point(M, q)
            @test is_vector(M, q, rand(M; vector_at = q))
            @test is_vector(M, q, rand(M; vector_at = q, tangent_distr = :Rician))
            @test is_vector(
                M, q, rand(MersenneTwister(123), M; vector_at = q, tangent_distr = :Rician)
            )
            # coordinates of a Gaussian tangent vector in the default orthonormal basis
            z = randn(MersenneTwister(42), 9)
            Z = rand(MersenneTwister(42), M; vector_at = p2, σ = 2.0)
            @test get_coordinates(M, p2, Z, basis) ≈ 2 * z
            # a random point drawn into a MatrixSqrtManifoldPoint stores its matrix square roots
            pR = rand!(MersenneTwister(42), M, MatrixSqrtManifoldPoint(p2))
            @test pR.sqrt ≈ MatrixSqrtManifoldPoint(pR.p).sqrt
            @test pR.sqrt_inv ≈ MatrixSqrtManifoldPoint(pR.p).sqrt_inv
        end
        @testset "Riemannian Hessian conversion" begin
            N = HermitianPositiveDefinite(2)
            p = ComplexF64[2 im; -im 2]
            G = ComplexF64[1 0; 1 + im 0.1]
            H = ComplexF64[2 1; 0 1im]
            Z = ComplexF64[0 3im; -3im 0]
            W = Z * 1 / 2 * (G' + G) * p
            W = 1 / 2 * p * (H' + H) * p + 1 / 2 * (W' + W)
            @test riemannian_Hessian(N, p, G, H, Z) == W
        end
    end

    @testset "MatrixSqrtManifoldPoint functions" begin
        p = MatrixSqrtManifoldPoint(2 * p1)
        @test MatrixSqrtManifoldPoint(p) === p
        @test ManifoldsBase.check_size(M, p; atol = 1.0e-8) === nothing
        @test ManifoldsBase.check_size(M, p, p1; atol = 1.0e-8) === nothing
        @test hash(p) == hash(MatrixSqrtManifoldPoint(2 * p1))
        pS = MatrixSqrtManifoldPoint(2 * p1; store_p = false, store_sqrt = false, store_sqrt_inv = false)
        @test convert(AbstractMatrix, pS) ≈ 2 * p1
        @test Manifolds.hpd_sqrt(pS) ≈ sqrt(2) * p1
        @test Manifolds.hpd_sqrt_inv(pS) ≈ p1 / sqrt(2)
        @test all(Manifolds.hpd_sqrt_and_sqrt_inv(pS) .≈ (sqrt(2) * p1, p1 / sqrt(2)))
        @test MatrixSqrtManifoldPoint(p3).sqrt ≈ MatrixSqrtManifoldPoint(p3).sqrt'
        @test MatrixSqrtManifoldPoint(p3).sqrt^2 ≈ p3
        # wrapped and static matrices
        @test MatrixSqrtManifoldPoint(Hermitian(p3)).sqrt ≈ MatrixSqrtManifoldPoint(p3).sqrt
        @test MatrixSqrtManifoldPoint(MMatrix{3, 3}(p3)).sqrt ≈ MatrixSqrtManifoldPoint(p3).sqrt
        q = exp(M, p, X)
        @test q isa MatrixSqrtManifoldPoint
        @test isapprox(M, q, exp(M, 2 * p1, X))
        @test eltype(q) == ComplexF64
    end
end
