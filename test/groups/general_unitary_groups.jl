include("../utils.jl")
include("group_utils.jl")

@testset "General Unitary Groups" begin
    @testset "Orthogonal Group" begin
        O2 = Orthogonal(2)
        @test repr(O2) == "Orthogonal(2)"
    end

    @testset "Unitary Group" begin
        U2 = Unitary(2)
        @test repr(U2) = "Unitary(2)"
    end

    @testset "Special Unitary Group" begin
        SU2 = SpecialUnitary(2)
        @test Repr(SU2) == "SpecialUnitary(2)"

        p = ones(2, 2)
        q = project(SU2, p)
        @test is_point(SU2, q, true)
        q2 = allocate(q)
        project!(SU2, a2, p)
        @test q == q2
    end
end
