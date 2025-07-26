include("../header.jl")

using RecursiveArrayTools

@testset "fiber bundle" begin
    M = Stiefel(3, 2)
    vm = default_vector_transport_method(M)
    @test Manifolds.FiberBundleProductVectorTransport(M) ==
        Manifolds.FiberBundleProductVectorTransport(vm, vm)
end
