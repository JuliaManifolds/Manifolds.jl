include("../header.jl")

@testset "Hamiltonian matrices" begin
    M = HamiltonianMatrices(4)
    Mf = HamiltonianMatrices(4; parameter = :field)
    # A has to be of the Form JS, S symmetric
    p =
        SymplecticElement(1.0) * [
        1.0 2.0 0.0 0.0
        2.0 3.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 4.0
    ]
    q =
        SymplecticElement(1.0) * [
        4.0 3.0 0.0 0.0
        3.0 2.0 0.0 0.0
        0.0 0.0 4.0 0.0
        0.0 0.0 0.0 1.0
    ]
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
        @test_throws DomainError is_point(M, ones(4, 4), true)
        @test_throws DomainError is_vector(M, p, ones(4, 4), true, true)
    end
    @testset "Basics" begin
        @test repr(M) == "HamiltonianMatrices(4, ℝ)"
        @test repr(Mf) == "HamiltonianMatrices(4, ℝ; parameter=:field)"
        @test is_point(M, p)
        @test is_point(M, Hamiltonian(p))
        @test is_vector(M, p, p)
        @test get_embedding(M) == Euclidean(4, 4; field = ℝ)
        @test get_embedding(Mf) == Euclidean(4, 4; field = ℝ, parameter = :field)
        @test embed(M, p) == p
        @test embed(M, p, p) == p
        @test is_flat(M)
        Random.seed!(42)
        is_point(M, rand(M))
    end
end
