include("../utils.jl")
include("group_utils.jl")

@testset "General Unitary Groups" begin
    @testset "Orthogonal Group" begin
        O2 = Orthogonal(2)
        @test repr(O2) == "Orthogonal(2)"

        for n in [2, 3, 4, 5] # 2-4 have special implementations, 5 for generic case
            On = Orthogonal(n)
            X = zeros(n, n)
            X[1, 2] = 1.0
            X[2, 1] = -1.0
            p = exp_lie(On, X)
            X2 = log_lie(On, p)
            e = Identity(MultiplicationOperation())
            # They are not yet inverting, p2 is on the “other half”
            # but taking the log again should also be X ahain
            if n != 4
                @test isapprox(On, e, X2, X)
            else
                @test_broken isapprox(On, e, X2, X)
            end
            @test log_lie(On, e) == zeros(n, n)
        end
    end

    @testset "Unitary Group" begin
        U2 = Unitary(2)
        @test repr(U2) == "Unitary(2)"
    end

    @testset "Special Unitary Group" begin
        SU2 = SpecialUnitary(2)
        @test repr(SU2) == "SpecialUnitary(2)"

        p = ones(2, 2)
        q = project(SU2, p)
        @test is_point(SU2, q, true)
        q2 = allocate(q)
        project!(SU2, q2, p)
        @test q == q2
        p2 = copy(p)
        p2[1, 1] = -1
        q2 = project(SU2, p2)
        @test is_point(SU2, q2, true)

        Xe = ones(2, 2)
        X = project(SU2, q, Xe)
        @test is_vector(SU2, q, X)
        @test_throws ManifoldDomainError is_vector(SU2, p, X, true, true) # base point wrong
        @test_throws DomainError is_vector(SU2, q, Xe, true, true) # Xe not skew hermitian
        @test_throws DomainError is_vector(
            SU2,
            Identity(AdditionOperation()),
            Xe,
            true,
            true,
        ) # base point wrong
        e = Identity(MultiplicationOperation())
        @test_throws DomainError is_vector(SU2, e, Xe, true, true) # Xe not skew hermitian
    end
end
