using LinearAlgebra, Manifolds, ManifoldsBase, Random, Test

@testset "Hamiltonian matrices" begin
    @test_throws DomainError HamiltonianMatrices(-2)
    M = HamiltonianMatrices(4)
    J = SymplecticElement(1.0)
    # Hamiltonian matrices are of the form JS with S symmetric
    p = J * [1.0 2.0 0.0 0.0; 2.0 3.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 4.0]
    q = J * [4.0 3.0 0.0 0.0; 3.0 2.0 0.0 0.0; 0.0 0.0 4.0 0.0; 0.0 0.0 0.0 1.0]
    r = J * [2.0 0.0 0.0 1.0; 0.0 1.0 1.0 0.0; 0.0 1.0 3.0 0.0; 1.0 0.0 0.0 2.0]
    X = q - p
    Y = J * [1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0]
    # the identity is its own symplectic inverse, so it is normal to the Hamiltonian matrices
    N = Matrix(1.0I, 4, 4)

    functions = [
        copy, copyto!, default_inverse_retraction_method, default_retraction_method,
        default_vector_transport_method, distance, embed, exp, flat, geodesic, get_embedding,
        injectivity_radius, inner, inverse_retract, is_flat, is_point, is_vector, log,
        manifold_dimension, mid_point, norm, parallel_transport_direction,
        parallel_transport_to, project, repr, representation_size, retract,
        sectional_curvature, sharp, shortest_geodesic, vector_transport_direction,
        vector_transport_to, Weingarten, zero_vector,
    ]
    common_expectations = Dict(
        :atol => 1.0e-12,
        default_inverse_retraction_method => LogarithmicInverseRetraction(),
        default_retraction_method => ExponentialRetraction(),
        default_vector_transport_method => ParallelTransport(),
        injectivity_radius => Inf,
        (injectivity_radius, ExponentialRetraction()) => Inf,
        is_flat => true,
        representation_size => (4, 4),
        sectional_curvature => 0.0,
        sectional_curvature_max => 0.0,
        sectional_curvature_min => 0.0,
        Weingarten => zeros(4, 4),
        :IsPointErrors => [DomainError],
        :IsVectorErrors => [DomainError],
    )

    @testset "Real Hamiltonian matrices" begin
        Manifolds.Test.test_manifold(
            M,
            Dict(
                :Covectors => [flat(M, p, X)],
                :EmbeddedPoints => [p + 2 * N],
                :EmbeddedVectors => [X + 3 * N],
                :Functions => [functions; rand],
                :InvalidPoints => [ones(4, 4)],
                :InvalidVectors => [ones(4, 4)],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :NormalVectors => [N],
                :Points => [p, q, r],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => Y,
                :Vectors => [X, r - q, p - r],
                :VectorTransportMethods => [ParallelTransport()],
                :Seed => 42,
            ),
            merge(
                common_expectations,
                Dict(
                    distance => norm(X),
                    (embed, :Point) => p,
                    (embed, :Vector) => X,
                    exp => q,
                    get_embedding => Euclidean(4, 4),
                    inner => dot(X, Y),
                    log => X,
                    manifold_dimension => 10,
                    mid_point => (p + q) / 2,
                    norm => norm(X),
                    parallel_transport_direction => X,
                    parallel_transport_to => X,
                    (project, :Point) => p,
                    (project, :Vector) => X,
                    repr => "HamiltonianMatrices(4, ℝ)",
                    sharp => X,
                    (vector_transport_to, ParallelTransport()) => X,
                    :IsVectorBasepointError => ManifoldDomainError,
                ),
            ),
        )
    end

    @testset "Complex Hamiltonian matrices" begin
        Mc = HamiltonianMatrices(4, ℂ)
        pc = p + im * r
        qc = q + im * p
        Xc = qc - pc
        Yc = Y + im * p
        Manifolds.Test.test_manifold(
            Mc,
            Dict(
                :Covectors => [flat(Mc, pc, Xc)],
                :EmbeddedPoints => [pc + (2 + im) * N],
                :EmbeddedVectors => [Xc + (3 - im) * N],
                :Functions => functions,
                :InvalidPoints => [complex(ones(4, 4))],
                :InvalidVectors => [complex(ones(4, 4))],
                :InverseRetractionMethods => [LogarithmicInverseRetraction()],
                :NormalVectors => [complex(N)],
                :Points => [pc, qc, complex(r)],
                :RetractionMethods => [ExponentialRetraction()],
                :SecondVector => Yc,
                :Vectors => [Xc, r - qc, pc - r],
                :VectorTransportMethods => [ParallelTransport()],
            ),
            merge(
                common_expectations,
                Dict(
                    distance => norm(Xc),
                    (embed, :Point) => pc,
                    (embed, :Vector) => Xc,
                    exp => qc,
                    get_embedding => Euclidean(4, 4; field = ℂ),
                    inner => dot(Xc, Yc),
                    log => Xc,
                    manifold_dimension => 20,
                    mid_point => (pc + qc) / 2,
                    norm => norm(Xc),
                    parallel_transport_direction => Xc,
                    parallel_transport_to => Xc,
                    (project, :Point) => pc,
                    (project, :Vector) => Xc,
                    repr => "HamiltonianMatrices(4, ℂ)",
                    sharp => Xc,
                    (vector_transport_to, ParallelTransport()) => Xc,
                    Weingarten => zeros(ComplexF64, 4, 4),
                    :IsVectorBasepointError => ManifoldDomainError,
                ),
            ),
        )
    end

    @testset "Hamiltonian" begin
        @test is_hamiltonian(p)
        pH = Hamiltonian(p)
        @test is_hamiltonian(pH)
        @test Hamiltonian(pH) === pH
        @test startswith("$(pH)", "Hamiltonian([")
        @test size(pH) == size(p)
        @test (pH^+).value == symplectic_inverse(p)
        @test symplectic_inverse(pH).value == symplectic_inverse(p)
        qH = Hamiltonian(q)
        pqH = pH * qH
        @test pqH isa Hamiltonian
        @test pqH.value == p * q
        pqH2 = pH + qH
        @test pqH2 isa Hamiltonian
        @test pqH2.value == p + q
        pqH3 = pH - qH
        @test pqH3 isa Hamiltonian
        @test pqH3.value == p - q
        @test is_point(M, pH)
    end

    @testset "Basics" begin
        @test riemann_tensor(M, p, p, q, p) == zeros(4, 4)
        @test sectional_curvature(M, p, p, q) === 0.0
        A_h = [1.0 -2.0 0.5 3.0; 4.0 1.5 -1.0 2.0; -3.0 0.25 2.0 -1.0; 0.5 5.0 -2.5 1.0]
        P_h = project(M, A_h)
        @test is_point(M, P_h)
        @test project(M, P_h) == P_h
        @test dot(A_h - P_h, p) == 0
        @test project(M, p, A_h) == P_h
        # the real and the imaginary part are projected separately
        @test project(HamiltonianMatrices(4, ℂ), A_h + im * p) == P_h + im * p
    end

    @testset "Random Hamiltonian matrices" begin
        q = rand(MersenneTwister(42), M)
        @test is_point(M, q)
        @test q[3:4, 1:2] != q[1:2, 3:4]
    end

    Manifolds.Test.test_manifold(
        HamiltonianMatrices(4; parameter = :field),
        Dict(:Functions => [get_embedding, is_point, manifold_dimension, repr], :Points => [p]),
        # Expectations
        Dict(
            get_embedding => Euclidean(4, 4; parameter = :field),
            manifold_dimension => 10,
            repr => "HamiltonianMatrices(4, ℝ; parameter=:field)",
        ),
    )
end
