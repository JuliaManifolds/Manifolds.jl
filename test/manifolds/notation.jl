include("utils.jl")

@testset "Test Notation" begin
    M = Sphere(2)
    p1 = [1.0, 0.0, 0.0]
    @test (p1 ∈ M) == is_point(M, p1)
    @test (2 * p1 ∈ M) == is_point(M, 2 * p1)
    X1 = [0.0, 1.0, 0.0]
    X2 = [1.0, 0.0, 0.0]
    TpM = TangentSpaceAtPoint(M, p1)
    @test (X1 ∈ TpM) == is_vector(M, p1, X1)
    @test (X2 ∈ TpM) == is_vector(M, p1, X2)
end
