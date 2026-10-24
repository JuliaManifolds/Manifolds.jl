using LinearAlgebra, Manifolds, ManifoldsBase, Random, StaticArrays, Test

@testset "Special unitary matrices" begin
    M = SpecialUnitaryMatrices(2)
    # tangent vectors are represented in the Lie algebra, i.e. as traceless skew-Hermitian matrices
    a = 0.5
    p = Matrix{ComplexF64}(I, 2, 2)
    q = ComplexF64[cos(a) sin(a); -sin(a) cos(a)]
    r = ComplexF64[1im 0.0; 0.0 -1im]
    X = ComplexF64[0.0 a; -a 0.0] # log(M, p, q)
    Y = ComplexF64[1im 0.5; -0.5 -1im]
    rot(t) = ComplexF64[cos(t) sin(t); -sin(t) cos(t)]
    B = DefaultOrthonormalBasis()

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Bases => [B],
            :Coordinates => [[1.0, 0.0, 0.0]],
            :Covectors => [flat(M, p, X)],
            # p and X with added normal parts, multiples of p and Hermitian matrices
            :EmbeddedPoints => [2 .* p],
            :EmbeddedVectors => [X .+ ComplexF64[1.0 2.0im; -2.0im 3.0] .+ 0.5im .* p],
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, distance, exp, flat, geodesic, get_basis,
                get_coordinates, get_embedding, get_vector, get_vectors, injectivity_radius,
                inner, inverse_retract, is_flat, is_point, is_vector, log, manifold_dimension,
                manifold_volume, mid_point, norm, parallel_transport_direction,
                parallel_transport_to, project, rand, repr, representation_size, retract,
                sectional_curvature, sharp, shortest_geodesic, vector_transport_direction,
                vector_transport_to, Weingarten, zero_vector,
            ],
            :InvalidPoints => [ComplexF64[1im 0.0; 0.0 1im], zeros(ComplexF64, 3, 3)],
            :InvalidVectors => [ComplexF64[1im 0.0; 0.0 1im], ones(ComplexF64, 2, 2)],
            :InverseRetractionMethods => [
                LogarithmicInverseRetraction(), missing,
            ],
            # multiples of p and Hermitian matrices are normal to M
            :NormalVectors => [3 .* p .+ 1im .* p],
            :Points => [p, q, r],
            :RetractionMethods => [ExponentialRetraction(), QRRetraction()],
            :SecondVector => Y,
            :Vectors => [X, log(M, q, r), log(M, r, p)],
            :VectorTransportMethods => [ParallelTransport(), ProjectionTransport()],
            :Seed => 42,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            :atols => Dict(geodesic => 1.0e-9, shortest_geodesic => 1.0e-9),
            default_inverse_retraction_method => LogarithmicInverseRetraction(),
            default_retraction_method => ExponentialRetraction(),
            default_vector_transport_method => ProjectionTransport(),
            distance => sqrt(2) * a,
            exp => q,
            get_embedding => Euclidean(2, 2; field = ℂ),
            (get_coordinates, B) => [sqrt(2) * a, 0.0, 0.0],
            (get_vector, [1.0, 0.0, 0.0], B) => ComplexF64[0.0 1.0; -1.0 0.0] / sqrt(2),
            (get_vectors, B) => :Orthonormal,
            injectivity_radius => π * sqrt(2),
            inner => dot(X, Y),
            is_flat => false,
            log => X,
            manifold_dimension => 3,
            manifold_volume => 4 * sqrt(2) * π^2,
            mid_point => ComplexF64[cos(a / 2) sin(a / 2); -sin(a / 2) cos(a / 2)],
            norm => sqrt(2) * a,
            (number_eltype, rand) => ComplexF64,
            # the velocity of a geodesic in the Lie algebra is constant
            parallel_transport_direction => X,
            parallel_transport_to => X,
            (project, :Point) => p,
            (project, :Vector) => X,
            (retract, ExponentialRetraction()) => q,
            repr => "SpecialUnitaryMatrices(2)",
            representation_size => (2, 2),
            # SU(2) has constant sectional curvature
            sectional_curvature => 0.5,
            sectional_curvature_max => 0.5,
            sectional_curvature_min => 0.5,
            sharp => X,
            (vector_transport_to, ParallelTransport()) => X,
            # the traceless skew-Hermitian part of qᴴpX = a(cos(a)J + sin(a)I) for X = aJ
            (vector_transport_to, ProjectionTransport()) => cos(a) .* X,
            # the eigenvalues of X are ±ia
            volume_density => sin(a)^2 / a^2,
            Weingarten => -3 .* X,
            :IsPointErrors => [DomainError, DomainError],
            :IsVectorErrors => [DomainError, DomainError],
            :IsVectorBasepointError => ManifoldDomainError,
        ),
    )

    @testset "Riemannian Hessian" begin
        # f(q) = Re tr(Aᴴq), whose second derivative along the geodesic in direction X is Re tr(Aᴴ p X²)
        A = ComplexF64[1.0 2.0im; -0.5 3.0]
        @test real(inner(M, p, riemannian_Hessian(M, p, A, zero(A), X), X)) ≈ real(tr(A' * p * X * X))
        # the Weingarten map applied to the normal part of the gradient
        @test riemannian_Hessian(M, p, A, zero(A), X) ≈ Weingarten(M, p, X, A - p * project(M, p, A))
    end

    @testset "sectional curvature" begin
        M3 = SpecialUnitaryMatrices(3)
        @test sectional_curvature_max(M3) == 0.5
        @test sectional_curvature_min(M3) == 0.0
        @test sectional_curvature_max(SpecialUnitaryMatrices(1)) == 0.0
    end

    @testset "volume, dimension and injectivity radius" begin
        @test manifold_volume(SpecialUnitaryMatrices(1)) ≈ 1
        @test manifold_volume(SpecialUnitaryMatrices(3)) ≈ 16 * sqrt(3) * π^5
        @test manifold_volume(SpecialUnitaryMatrices(4)) ≈ 256 * π^9 / 3
        @test manifold_dimension(SpecialUnitaryMatrices(3)) == 8
        @test manifold_dimension(SpecialUnitaryMatrices(4)) == 15
        @test injectivity_radius(SpecialUnitaryMatrices(3)) == π * sqrt(2)
        @test injectivity_radius(SpecialUnitaryMatrices(4)) == π * sqrt(2)
    end

    @testset "traceless tangent vectors" begin
        M3 = SpecialUnitaryMatrices(3)
        p3 = Matrix{ComplexF64}(I, 3, 3)
        @test !is_vector(M3, p3, 1.0im * Matrix{ComplexF64}(I, 3, 3))
        @test !is_vector(M3, p3, 1.0e-9im * Matrix{ComplexF64}(I, 3, 3))
        Z = ComplexF64[1.0 0.2im -0.3; 0.4 -0.5im 0.6; 0.7 0.8 -0.9im]
        X3 = project(M3, p3, Z)
        # the skew-Hermitian part of Z has the trace -1.4im
        @test X3 ≈ (Z - Z') / 2 + 1.4im / 3 * I
        @test is_vector(M3, p3, X3)
        @test is_point(M3, exp(M3, p3, X3))
        @test is_vector(M3, p3, project(M3, p3, 1.0e9 * Z))
        # skew-Hermitian up to 1e-7, so accepted only with a larger tolerance
        Y3 = ComplexF64[1.0e-7 1.0 0.0; -1.0 -1.0e-7 0.0; 0.0 0.0 0.0]
        @test !is_vector(M3, p3, Y3)
        @test is_vector(M3, p3, Y3; atol = 1.0e-6)
    end

    @testset "rand" begin
        M3 = SpecialUnitaryMatrices(3)
        p3 = rand(MersenneTwister(4711), M3)
        @test is_point(M3, p3)
        @test det(p3) ≈ 1
        X3 = rand(MersenneTwister(4711), M3; vector_at = p3)
        @test is_vector(M3, p3, X3)
        @test is_point(M3, exp(M3, p3, X3))
        q3 = similar(p3)
        rand!(MersenneTwister(4711), M3, q3)
        @test is_point(M3, q3)
    end

    @testset "logarithm of minimal norm" begin
        M3 = SpecialUnitaryMatrices(3)
        p3 = Matrix{ComplexF64}(I, 3, 3)
        q3 = Matrix{ComplexF64}(Diagonal([cis(1.6), cis(1.6), cis(-3.2)]))
        X3 = log(M3, p3, q3)
        @test is_vector(M3, p3, X3)
        @test sort(imag.(eigvals(X3))) ≈ [-3.2, 1.6, 1.6]
        @test isapprox(M3, exp(M3, p3, X3), q3)
        @test distance(M3, p3, q3) ≈ norm([1.6, 1.6, -3.2])
        W = [0.0 0.5im 0.2; -0.2 0.3im 0.1im; 0.0 0.1im -0.3im]
        r3 = exp(M3, p3, W - W')
        @test isapprox(M3, exp(M3, r3, log(M3, r3, q3 * r3)), q3 * r3)
        @test distance(M3, r3, q3 * r3) ≈ norm([1.6, 1.6, -3.2])
        Z = log(M3, p3, q3')
        @test sort(imag.(eigvals(Z))) ≈ [-1.6, -1.6, 3.2]
        @test log(M3, SMatrix{3, 3}(p3), SMatrix{3, 3}(q3)) ≈ X3
    end

    @test repr(SpecialUnitaryMatrices(3; parameter = :field)) ==
        "SpecialUnitaryMatrices(3; parameter=:field)"
end
