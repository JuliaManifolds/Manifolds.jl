include("utils.jl")
using LightGraphs, SimpleWeightedGraphs

@testset "Graph Manifold" begin
    @testset "Plain Graph" begin
        G = SimpleGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)
        M = Euclidean(2)
        N = GraphManifold(G,M,VertexManifold())
        @test manifold_dimension(N) == manifold_dimension(M)*nv(G)
        @test manifold_dimension(GraphManifold(G,M,EdgeManifold())) == manifold_dimension(M)*ne(G)
        x = [[1,2], [2,3], [2,3]]
        y = [[4,5], [6,7], [8,9]]
        z = [[6,5], [4,3], [2,8]]
        pts = [x,y,z]
        test_manifold(N, pts)
    end
end
