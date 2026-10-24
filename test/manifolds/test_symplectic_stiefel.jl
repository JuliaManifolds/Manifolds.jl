using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test
using ADTypes, FiniteDifferences, ManifoldDiff
using Manifolds: RiemannianProjectionBackend

@testset "Symplectic Stiefel" begin
    @test_throws DomainError SymplecticStiefel(2, 4)
    @test_throws ArgumentError SymplecticStiefel(5, 4)
    @test_throws ArgumentError SymplecticStiefel(6, 3)

    M = SymplecticStiefel(4, 2)
    J = SymplecticElement(1.0)
    # the orthosymplectic matrix [e₁ e₃] rotated by the angle a towards [e₂ e₄]
    a = 0.5
    p = [1.0 0.0; 0.0 0.0; 0.0 1.0; 0.0 0.0]
    q = [cos(a) 0.0; sin(a) 0.0; 0.0 cos(a); 0.0 sin(a)]
    # a point with r'r = diag(4, 1/4)
    r = [2.0 0.0; 0.0 0.0; 0.0 0.5; 0.0 0.0]
    E = [0.0 0.0; 1.0 0.0; 0.0 0.0; 0.0 1.0]
    X = a .* E
    # a Hamiltonian 2×2 matrix Ω, so that pΩ is a tangent vector
    Ω = [1.0 0.0; 0.0 -1.0]
    Y = [0.0 0.0; 1.0 2.0; 0.0 0.0; 3.0 1.0] + p * Ω
    # the velocity of the rotation at q
    Xq = a .* [-sin(a) 0.0; cos(a) 0.0; 0.0 -sin(a); 0.0 cos(a)]

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, embed, exp, geodesic, get_embedding, inner,
                inverse_retract, is_flat, is_point, is_vector, manifold_dimension, norm, rand,
                repr, representation_size, retract, vector_transport_to, zero_vector,
            ],
            :InvalidPoints => [2 .* p],
            :InvalidVectors => [X + p],
            :InverseRetractionMethods => [CayleyInverseRetraction(), missing],
            :Points => [p, q, r],
            :RetractionMethods => [CayleyRetraction(), ExponentialRetraction()],
            :SecondVector => Y,
            :Vectors => [X, Xq, E],
            :VectorTransportMethods => [ProjectionTransport()],
            :Seed => 42,
            :TestRandScaling => true,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            :atols => Dict(geodesic => 1.0e-9),
            default_inverse_retraction_method => CayleyInverseRetraction(),
            default_retraction_method => CayleyRetraction(),
            default_vector_transport_method => ProjectionTransport(),
            (embed, :Point) => p,
            (embed, :Vector) => X,
            # the rotation is a geodesic
            exp => q,
            get_embedding => Euclidean(4, 2),
            # pᵀJX = 0 and pᵀp = I, so the metric at p reduces to the Euclidean one
            inner => dot(X, Y),
            # the Cayley transform of the rotation by a is (1 - a²/4, a)/(1 + a²/4), so the
            # inverse Cayley retraction of q is the tangent vector 2tan(a/2)E
            (inverse_retract, CayleyInverseRetraction()) => 2 * tan(a / 2) .* E,
            is_flat => false,
            manifold_dimension => 7,
            norm => norm(X),
            repr => "SymplecticStiefel(4, 2)",
            representation_size => (4, 2),
            (retract, CayleyRetraction()) =>
                [1 - a^2 / 4 0.0; a 0.0; 0.0 1 - a^2 / 4; 0.0 a] ./ (1 + a^2 / 4),
            (retract, ExponentialRetraction()) => q,
            # q is orthosymplectic, and the projection removes the part a sin(a)q of X
            (vector_transport_to, ProjectionTransport()) => cos(a) .* Xq,
            :IsPointErrors => [DomainError],
            :IsVectorErrors => [DomainError],
        ),
    )

    @testset "Basics" begin
        @test base_manifold(M) === M
        @test get_total_space(M) == SymplecticMatrices(4)
        @test metric(M) == RealSymplecticMetric()
        @test manifold_dimension(SymplecticStiefel(6, 4)) == 18
        # at r the metric weighs the columns by (r'r)⁻¹ = diag(1/4, 4)
        @test inner(M, r, E, E) ≈ 1 / 4 + 4
        # vertical tangent vectors pΩ have the norm ‖Ω‖/√2
        @test norm(M, p, p * Ω) ≈ 1
        @test inner(M, Float32.(p), Float32.(X), Float32.(X)) isa Float32
        @test ManifoldsBase.retract_fused(M, p, E, a, CayleyRetraction()) ≈
            [1 - a^2 / 4 0.0; a 0.0; 0.0 1 - a^2 / 4; 0.0 a] ./ (1 + a^2 / 4)
        # the rounding errors of p⁺p grow with the norm of a point, here the columns 1, 2, 4
        # and 5 of a symplectic matrix, a product of two shears with symmetric blocks A and C
        # and of a scaling with an invertible B
        A = [20 * sqrt(i + j) for i in 1:3, j in 1:3]
        C = [20 * sqrt(i * j + 1) for i in 1:3, j in 1:3]
        B = [2.0 1 / 3 1 / 3; 0.0 4.0 1 / 3; 0.0 0.0 8.0]
        Z = zeros(3, 3)
        p_big = ([I A; Z I] * [B Z; Z inv(B)'] * [I Z; C I])[:, [1, 2, 4, 5]]
        @test norm(p_big) > 100
        @test norm(inv(SymplecticStiefel(6, 4), p_big) * p_big - I) > 1.0e-12
        @test is_point(SymplecticStiefel(6, 4), p_big; atol = 1.0e-14)
        @test !is_point(SymplecticStiefel(6, 4), p_big .+ 1.0e-3; atol = 1.0e-14)
    end

    @testset "Symplectic inverse" begin
        r_inv = [0.5 0.0 0.0 0.0; 0.0 0.0 2.0 0.0]
        @test inv(M, r) == r_inv
        @test inv!(M, zeros(2, 4), r) == r_inv
        @test Manifolds.symplectic_inverse_times(M, r, q) == r_inv * q
        @test Manifolds.symplectic_inverse_times!(M, zeros(2, 2), r, q) == r_inv * q
        @test Manifolds.symplectic_inverse_times(M, r, r) == I
    end

    @testset "Projection and canonical projection" begin
        # matrices JpW with a skew-symmetric W are normal to the tangent space at p
        W = [0.0 1.0; -1.0 0.0]
        @test project(M, p, X + J * p * W) ≈ X
        Z = similar(X)
        project!(M, Z, p, X + J * p * W)
        @test Z ≈ X
        A = [-7.0 2.0; 4.0 0.0; -1.0 -1.0; -18.0 4.0]
        @test is_vector(M, r, project(M, r, A); error = :error)
        # the canonical projection takes the columns one and n + 1 of a symplectic matrix
        P = Matrix(Diagonal([2.0, 1.0, 0.5, 1.0]))
        @test canonical_project(M, P) == r
        Z = similar(r)
        canonical_project!(M, Z, P)
        @test Z == r
    end

    @testset "Exponential map" begin
        # the definition of the exponential map, evaluated directly
        function exp_naive(p, X)
            pT_p = lu(p' * p)
            X_inv_pTp_pT = X * (pT_p \ p')
            Ωp = J * X_inv_pTp_pT' * (I - J' * p * (pT_p \ p') * J) * J + X_inv_pTp_pT
            return exp(Ωp - Ωp') * exp(Array(Ωp')) * p
        end
        Z = E + r * Ω
        @test exp(M, r, Z) ≈ exp_naive(r, Z)
    end

    @testset "Gradients" begin
        G = [1.0 2.0; 0.5 -1.0; 3.0 0.0; -2.0 1.0]
        Z = riemannian_gradient(M, r, G)
        @test riemannian_gradient(M, r, G; embedding_metric = EuclideanMetric()) == Z
        Z2 = similar(Z)
        riemannian_gradient!(M, Z2, r, G)
        @test Z2 ≈ Z
        @test is_vector(M, r, Z)
        # Z is the Riesz representer of the differential, here evaluated on tangent vectors
        for V in (E, r * Ω, r * [0.0 1.0; 1.0 0.0])
            @test inner(M, r, Z, V) ≈ dot(G, V)
        end
        # the trace of the upper k×k block of a point
        f(p) = tr(p[1:1, 1:1]) + p[3, 2]
        fd = RiemannianProjectionBackend(AutoFiniteDifferences(central_fdm(5, 1)))
        Gf = [1.0 0.0; 0.0 0.0; 0.0 1.0; 0.0 0.0]
        @test isapprox(Manifolds.gradient(M, f, r, fd), riemannian_gradient(M, r, Gf); atol = 1.0e-9)
        Manifolds.gradient!(M, f, Z2, r, fd)
        @test isapprox(Z2, riemannian_gradient(M, r, Gf); atol = 1.0e-9)
    end

    Manifolds.Test.test_manifold(
        SymplecticStiefel(4, 2; parameter = :field),
        Dict(
            :Functions => [get_embedding, is_point, is_vector, manifold_dimension, repr],
            :Points => [p], :Vectors => [X],
        ),
        # Expectations
        Dict(
            get_embedding => Euclidean(4, 2; parameter = :field),
            manifold_dimension => 7,
            repr => "SymplecticStiefel(4, 2; parameter=:field)",
        ),
    )
    @test get_total_space(SymplecticStiefel(4, 2; parameter = :field)) ==
        SymplecticMatrices(4; parameter = :field)
end
