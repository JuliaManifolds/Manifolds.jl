include("utils.jl")

@testset "Maps" begin
    r2 = Euclidean(2)
    s2 = Sphere(2)
    m1 = FunctionMap(r2, s2) do x
        return [sin(x[1]) * cos(x[2]), sin(x[1]) * sin(x[2]), cos(x[1])]
    end

    @test domain(m1) == r2
    @test codomain(m1) == s2
    @test isapprox(s2, m1([0.0, 0.0]), [0.0, 0.0, 1.0])


end
