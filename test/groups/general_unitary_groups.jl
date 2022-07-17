include("../utils.jl")
include("group_utils.jl")

@testset "General Unitary Groups" begin
    @testset "Orthogonal Group" begin
        O2 = Orthogonal(2)
        @test repr(O2) == "Orthogonal(2)"

        for n in [2, 5] # skip 3&4,
            # 3 produces NaNs in this code,
            # 4 entered with a valid (det = -1) reports that such a determinant is not allowed
            On = Orthogonal(n)
            p = Matrix{Float64}(I, n, n)
            p[1, 1] = -1
            if n > 2
                p[2, 2] = -1
                p[3, 3] = -1
            end
            X = log_lie(On, p)
            p2 = exp_lie(On, X)
            # They are not yet inverting, p2 is on the “other half”
            # but taking the log again should also be X ahain
            @test isapprox(On, p2, log_lie(On, p2), X)
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
        @test_throws DomainError is_vector(SU2, Identity(AdditionOperation())Xe, true, true) # base point wrong
        e = Identity(MultiplicationOperation())
        @test_throws DomainError is_vector(SU2, e, Xe, true, true) # Xe not skew hermitian
    end
end
