using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test
using ADTypes, FiniteDifferences, ManifoldDiff
using Manifolds: RiemannianProjectionBackend

@testset "Symplectic matrices" begin
    @test_throws DomainError SymplecticMatrices(-2)
    @test_throws ArgumentError SymplecticMatrices(3)

    M = SymplecticMatrices(4)
    J = SymplecticElement(1.0)
    p = Matrix(Diagonal([2.0, 1.0, 0.5, 1.0]))
    # p⁻¹X = H is symmetric and Hamiltonian, so exp_p(X) = p exp(H)
    h = [1.0, -0.5, -1.0, 0.5]
    H = Matrix(Diagonal(h))
    X = p * H
    q = p * Diagonal(exp.(h))
    # [I B; 0 I] is symplectic for a symmetric B
    r = [1.0 0.0 1.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    # a Hamiltonian matrix [A B; C -Aᵀ] with symmetric B and C
    H2 = [1.0 1.0 1.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 -1.0 0.0; 0.0 1.0 -1.0 0.0]
    Y = p * H2
    # on diagonal matrices the projection onto the tangent space at the diagonal q decouples
    # into the projections of (Xᵢᵢ, Xᵢ₊₂ᵢ₊₂) onto the lines spanned by (qᵢᵢ, -1/qᵢᵢ)
    qd, Xd = diag(q)[1:2], diag(X)
    c = (qd .* Xd[1:2] .- Xd[3:4] ./ qd) ./ (qd .^ 2 .+ qd .^ -2)
    Xq = Matrix(Diagonal([c .* qd; -c ./ qd]))

    Manifolds.Test.test_manifold(
        M,
        Dict(
            :Functions => [
                copy, copyto!, default_inverse_retraction_method, default_retraction_method,
                default_vector_transport_method, distance, embed, exp, flat, geodesic,
                get_embedding, inner, inverse_retract, is_flat, is_point, is_vector,
                manifold_dimension, norm, rand, repr, representation_size, retract, sharp,
                vector_transport_to, zero_vector,
            ],
            :Covectors => [flat(M, p, X)],
            :InvalidPoints => [2 .* p],
            :InvalidVectors => [ones(4, 4)],
            :InverseRetractionMethods => [CayleyInverseRetraction(), missing],
            :Points => [p, q, r],
            :RetractionMethods => [CayleyRetraction(), ExponentialRetraction()],
            :SecondVector => Y,
            :Vectors => [X, q * H2, r * H2],
            :VectorTransportMethods => [ProjectionTransport()],
            :Seed => 42,
        ),
        # Expectations
        Dict(
            :atol => 1.0e-14,
            :atols => Dict(geodesic => 1.0e-9),
            default_inverse_retraction_method => CayleyInverseRetraction(),
            default_retraction_method => CayleyRetraction(),
            default_vector_transport_method => ProjectionTransport(),
            # the matrix logarithm of p⁺q = exp(H) is H
            distance => norm(H),
            (embed, :Point) => p,
            (embed, :Vector) => X,
            exp => q,
            get_embedding => Euclidean(4, 4),
            inner => dot(H, H2),
            # q = p exp(H) and p⁻¹X = 2 tanh(H/2) solve the Cayley retraction p(2I - H)⁻¹(2I + H) = q
            (inverse_retract, CayleyInverseRetraction()) => p * Diagonal(2 .* tanh.(h ./ 2)),
            is_flat => false,
            manifold_dimension => 10,
            norm => norm(H),
            repr => "SymplecticMatrices(4, ℝ)",
            representation_size => (4, 4),
            (retract, CayleyRetraction()) => p * Diagonal((2 .+ h) ./ (2 .- h)),
            (retract, ExponentialRetraction()) => q,
            sharp => X,
            (vector_transport_to, ProjectionTransport()) => Xq,
            :IsPointErrors => [DomainError],
            :IsVectorErrors => [DomainError],
        ),
    )

    @testset "Basics" begin
        @test metric(M) == RealSymplecticMetric()
        @test inner(MetricManifold(M, RealSymplecticMetric()), p, X, Y) == inner(M, p, X, Y)
        # p⁻¹X/2 = H/2, so the Cayley retraction is p(2I - H/2)⁻¹(2I + H/2)
        @test ManifoldsBase.retract_fused(M, p, X, 0.5, CayleyRetraction()) ≈
            p * Diagonal((4 .+ h) ./ (4 .- h))
        # the rounding errors of p⁺p grow with the norm of a point, here a product of
        # two shears with symmetric blocks A and C and of a scaling with an invertible B
        A = [10 * sqrt(i + j) for i in 1:2, j in 1:2]
        C = [10 * sqrt(i * j + 1) for i in 1:2, j in 1:2]
        B = [2.0 1 / 3; 0.0 4.0]
        Z = zeros(2, 2)
        p_big = [I A; Z I] * [B Z; Z inv(B)'] * [I Z; C I]
        @test norm(p_big) > 100
        @test norm(inv(M, p_big) * p_big - I) > 1.0e-13
        @test is_point(M, p_big; error = :error)
        @test !is_point(M, p_big .+ 1.0e-9)
    end

    @testset "Symplectic inverse" begin
        p_inv = Matrix(Diagonal([0.5, 1.0, 2.0, 1.0]))
        r_inv = [1.0 0.0 -1.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
        @test inv(M, p) == p_inv
        @test inv(M, r) == r_inv
        @test symplectic_inverse(r) == r_inv
        r2 = copy(M, r)
        inv!(M, r2)
        @test r2 == r_inv
        @test Manifolds.symplectic_inverse_times(M, r, q) == r_inv * q
        A = similar(q)
        Manifolds.symplectic_inverse_times!(M, A, r, q)
        @test A == r_inv * q
        @test Manifolds.symplectic_inverse_times(M, r, r) == I
    end

    @testset "Projections" begin
        # matrices p⁻ᵀJW with a skew-symmetric W are normal to the tangent space at p
        W = [0.0 1.0 2.0 0.0; -1.0 0.0 0.0 3.0; -2.0 0.0 0.0 1.0; 0.0 -3.0 -1.0 0.0]
        N = p' \ (J * W)
        @test project(M, p, X + N) ≈ X
        Z = similar(X)
        project!(M, Z, p, X + N)
        @test Z ≈ X
        # with respect to the extended metric, pJW is normal to the tangent space at p
        Mext = MetricManifold(get_embedding(M), ExtendedSymplecticMetric())
        NJ = p * J * W
        @test project(Mext, p, X + NJ) ≈ X
        @test Manifolds.project_normal!(Mext, Z, p, X + NJ) ≈ NJ
        @test Manifolds.project_normal!(Mext, Z, p, X) ≈ zeros(4, 4) atol = 1.0e-15
    end

    @testset "Change of representer and gradients" begin
        A = [5.0 -21.5 1.0 0.0; 3.14 14.9 0.0 2.0; 0.0 1.0 2.0 -1.0; 1.0 0.0 3.0 4.0]
        # g_p(c_p(A), Z) = tr(AᵀZ) for all tangent vectors Z
        Mext = MetricManifold(get_embedding(M), ExtendedSymplecticMetric())
        for cA in (
                change_representer(M, EuclideanMetric(), r, A),
                change_representer(Mext, EuclideanMetric(), r, A),
            )
            @test inner(M, r, cA, r * H) ≈ dot(A, r * H)
            @test inner(M, r, cA, r * H2) ≈ dot(A, r * H2)
        end
        @test is_vector(M, r, change_representer(M, EuclideanMetric(), r, A))
        B = similar(A)
        change_representer!(Mext, B, EuclideanMetric(), r, A)
        @test B ≈ r * r' * A
        # the Euclidean gradient of tr is the identity
        grad_tr = (r * r' + r * J * r * J) / 2
        @test riemannian_gradient(M, r, one(r)) ≈ grad_tr
        riemannian_gradient!(M, B, r, one(r))
        @test B ≈ grad_tr
        @test riemannian_gradient(M, p, one(p)) ≈ Diagonal([1.5, 0.0, -0.375, 0.0])
        fd = RiemannianProjectionBackend(AutoFiniteDifferences(central_fdm(5, 1)))
        for extended_metric in (true, false)
            @test isapprox(
                Manifolds.gradient(M, tr, r, fd; extended_metric = extended_metric), grad_tr;
                atol = 1.0e-9,
            )
            Manifolds.gradient!(M, tr, B, r, fd; extended_metric = extended_metric)
            @test isapprox(B, grad_tr; atol = 1.0e-9)
        end
    end

    @testset "SymplecticElement" begin
        @test SymplecticElement() == SymplecticElement(1)
        pQ_1 = [
            0.0 0 -2.0 3.0
            0.0 0.0 1.0 -1.0
            -1.0 -1.0 0.0 0.0
            -3.0 -2.0 4.0 -4.0
        ]
        pQ_2 = [
            0.0 0.0 -2.0 5.0
            0.0 0.0 1.0 -2.0
            -2.0 -1.0 0.0 0.0
            -5.0 -2.0 4 -8.0
        ]
        p_odd_row = [
            0.0 0.0 -1.0 3.0
            0.0 0.0 1.0 0.0
            -2.0 -1.0 0.0 0.0
        ]
        p_odd_col = [
            0.0 -2.0 5.0
            0.0 1.0 -2.0
            -1.0 0.0 0.0
            -2.0 4.0 -8.0
        ]
        Q = SymplecticElement(pQ_1, pQ_2)
        Q2 = SymplecticElement(1.0)

        @testset "Type Basics" begin
            @test Q == Q2
            @test ndims(Q) == 2
            @test copy(Q) == Q
            @test eltype(SymplecticElement(1 // 1)) == Rational{Int64}
            @test convert(SymplecticElement{Float64}, Q) == SymplecticElement(1.0)
            @test "$Q" == "SymplecticElement{Float64}(): 1.0*[0 I; -I 0]"
            @test "$(SymplecticElement(1 + 2im))" ==
                "SymplecticElement{Complex{Int64}}(): (1 + 2im)*[0 I; -I 0]"
        end

        @testset "Matrix Operations" begin
            @test -Q == SymplecticElement(-1.0)
            @test (2 * Q) * (5 / 6) == SymplecticElement(5 / 3)

            @testset "Powers" begin
                @test inv(Q) * Q == I
                @test inv(SymplecticElement(-4.0 + 8im)) * SymplecticElement(-4.0 + 8im) ==
                    UniformScaling(1.0 + 0.0im)
                @test Q * Q == -I
                @test Q^2 == -I
                @test Q^3 == -Q
                @test Q^4 == I
            end
            @testset "Addition (subtraction)" begin
                @test Q + Q == 2 * Q
                @test Q - SymplecticElement(1.0) == SymplecticElement(0.0)
                @test Q + pQ_1 == [
                    0.0 0.0 -1.0 3.0
                    0.0 0.0 1.0 0.0
                    -2.0 -1.0 0.0 0.0
                    -3.0 -3.0 4.0 -4.0
                ]
                @test Q - pQ_1 == [
                    0.0 0.0 3.0 -3.0
                    0.0 0.0 -1.0 2.0
                    0.0 1.0 0.0 0.0
                    3.0 1.0 -4.0 4.0
                ]
                @test pQ_1 - Q == [
                    0.0 0.0 -3.0 3.0
                    0.0 0.0 1.0 -2.0
                    0.0 -1.0 0.0 0.0
                    -3.0 -1.0 4.0 -4.0
                ]
                @test (pQ_1 + Q) == (Q + pQ_1)
                @test_throws ArgumentError Q + p_odd_row
            end
            @testset "Transpose-Adjoint" begin
                @test Q' == SymplecticElement(-1.0)
                @test transpose(SymplecticElement(10.0)) == SymplecticElement(-10.0)
                @test transpose(SymplecticElement(1 - 2.0im)) == SymplecticElement(-1 + 2.0im)
                @test adjoint(Q) == -Q
                @test adjoint(SymplecticElement(1 - 2.0im)) == SymplecticElement(-1 - 2.0im)
                @test adjoint(SymplecticElement(-1im)) == SymplecticElement(-1im)
                @test adjoint(SymplecticElement(2.0)) == SymplecticElement(-2.0)
            end
            @testset "Inplace mul!" begin
                z1 = [1 + 2im; 1 - 2im]
                @test lmul!(Q, copy(z1)) == Q * z1
                @test lmul!(Q, copy(p_odd_col)) == [
                    -1 0 0.0
                    -2 4 -8.0
                    0 2 -5.0
                    0 -1 2.0
                ]
                @test_throws ArgumentError lmul!(Q, copy(p_odd_row))

                @test rmul!(copy(z1'), Q) == z1' * Q
                @test rmul!(copy(p_odd_row), Q) == [
                    1 -3 0 0.0
                    -1 0 0 0.0
                    0 0 -2 -1.0
                ]
                @test_throws ArgumentError rmul!(copy(p_odd_col), Q)

                z1_copy = copy(z1)
                mul!(z1_copy, Q, z1)
                @test z1_copy == Q * z1

                @test_throws ArgumentError mul!(copy(p_odd_row), Q, p_odd_row)
                @test_throws ArgumentError mul!(copy(p_odd_row), p_odd_col, Q)

                @test_throws ArgumentError Q * p_odd_row
                @test_throws ArgumentError p_odd_col * Q

                mul!(z1_copy', z1', Q)
                @test z1_copy' == (z1' * Q)
            end
        end

        @testset "Symplectic inverse" begin
            @test (Q' * pQ_1' * Q) * pQ_1 == I
            @test is_point(M, pQ_1)
            @test is_point(M, pQ_2)
        end
    end

    Manifolds.Test.test_manifold(
        SymplecticMatrices(4; parameter = :field),
        Dict(
            :Functions => [get_embedding, is_point, is_vector, manifold_dimension, repr],
            :Points => [p], :Vectors => [X],
        ),
        # Expectations
        Dict(
            get_embedding => Euclidean(4, 4; parameter = :field),
            manifold_dimension => 10,
            repr => "SymplecticMatrices(4, ℝ; parameter=:field)",
        ),
    )
end
