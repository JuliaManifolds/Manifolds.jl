include("utils.jl")

@testset "Median and Mean" begin
    M = Sphere(2)
    p = [0.,0.,1.]
    n=8
    x = [ exp(M,p,2/n*[cos(α), sin(α), 0.]) for α = range(0,2*π - 2*π/n, length=n) ]
    y = mean(M,x)
    @test is_manifold_point(M,y)
    @test isapprox(M,y,p; atol=10^-7)
    z = median(M,x)
    z2 = median(M,x; use_random=true)
    @test is_manifold_point(M,z)
    @test isapprox(M,z,z2; atol=5*10^-5)
end