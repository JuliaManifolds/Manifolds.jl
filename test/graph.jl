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
        x = [1. 2. 3.;4. 5. 6.]
        y = [4. 6. 8.;5. 7. 9.]
        z = [6. 4. 2.;5. 3. 8.]
        pts = [x,y,z]
        test_manifold(N, pts)
    end
end
