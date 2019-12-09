include("utils.jl")

@testset "Median and Mean" begin
    M = Sphere(2)
    p = [0.,0.,1.]
    n=3
    x = [ exp(M,p,2/n*[cos(α), sin(α), 0.]) for α = range(0,2*π - 2*π/n, length=n) ]
    y = mean(M,x)
    @test is_manifold_point(M,y; atol=10^-9)
    @test isapprox(M,y,p; atol=10^-7)
    z = median(M,x; atol=10^-10)
    z2 = median(M,x; use_random=true, atol=10^-10)
    @test is_manifold_point(M,z)
    @test isapprox(M,z,z2; atol=5*10^-5)

    x = fill([0.,0.,1.],5)
    @test var(M,x) == 0.
    
    x = [ [1., 0., 0.], [0., 1., 0.] ]
    @test isapprox(M, mean(M,x), geodesic(M,x[1],x[2],π/4))
    @test var(M,x) == var(M,x,true,mean(M,x))
    @test var(M,x) ≈ 2*(π/4)^2
end