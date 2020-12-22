include("utils.jl")

@testset "Graph Manifold" begin
    M = Euclidean(2)
    x = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    y = [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]
    z = [[6.0, 5.0], [4.0, 3.0], [2.0, 8.0]]
    @testset "Plain Graph" begin
        G = SimpleGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)
        N = GraphManifold(G, M, VertexManifold())
        @test manifold_dimension(N) == manifold_dimension(M) * nv(G)
        @test manifold_dimension(GraphManifold(G, M, EdgeManifold())) ==
              manifold_dimension(M) * ne(G)
        @test is_manifold_point(N, x)
        @test !is_manifold_point(N, [x..., [0.0, 0.0]]) # an entry too much
        @test_throws DomainError is_manifold_point(N, [x..., [0.0, 0.0]], true)
        @test is_tangent_vector(N, x, log(N, x, y))
        @test !is_tangent_vector(N, x[1:2], log(N, x, y))
        @test_throws DomainError is_tangent_vector(N, x[1:2], log(N, x, y), true)
        @test !is_tangent_vector(N, x[1:2], log(N, x, y)[1:2])
        @test_throws DomainError is_tangent_vector(N, x, log(N, x, y)[1:2], true)
        @test incident_log(N, x) == [x[2] - x[1], x[1] - x[2] + x[3] - x[2], x[2] - x[3]]

        pts = [x, y, z]
        test_manifold(
            N,
            pts;
            test_representation_size=false,
            test_reverse_diff=false, #VERSION > v"1.2",
        )
        @test sprint(show, "text/plain", N) == """
        GraphManifold
        Graph:
         {3, 2} undirected simple Int64 graph
        Manifold on vertices:
         Euclidean(2; field = ℝ)"""

        NE = GraphManifold(G, M, EdgeManifold())
        @test is_manifold_point(NE, x[1:2])
        @test !is_manifold_point(NE, x) # an entry too much
        @test_throws DomainError is_manifold_point(NE, x, true)
        @test is_tangent_vector(NE, x[1:2], log(N, x, y)[1:2])
        @test !is_tangent_vector(NE, x, log(N, x, y))
        @test_throws DomainError is_tangent_vector(NE, x, log(N, x, y), true)
        @test !is_tangent_vector(N, x[1:2], log(N, x, y))
        @test_throws DomainError is_tangent_vector(NE, x[1:2], log(N, x, y), true)

        test_manifold(
            NE,
            [x[1:2], y[1:2], z[1:2]];
            test_representation_size=false,
            test_reverse_diff=false, #VERSION > v"1.2",
        )
        @test sprint(show, "text/plain", NE) == """
        GraphManifold
        Graph:
         {3, 2} undirected simple Int64 graph
        Manifold on edges:
         Euclidean(2; field = ℝ)"""

        G2 = SimpleDiGraph(3)
        add_edge!(G2, 1, 2)
        add_edge!(G2, 2, 3)
        N2 = GraphManifold(G2, M, VertexManifold())
        @test incident_log(N2, x) == [x[2] - x[1], x[3] - x[2], [0.0, 0.0]]
    end
    @testset "Weighted Graph" begin
        G3 = SimpleWeightedGraph(3)
        add_edge!(G3, 1, 2, 1.5)
        add_edge!(G3, 2, 3, 0.5)
        N3 = GraphManifold(G3, M, VertexManifold())
        @test incident_log(N3, x) == [
            1.5 * (x[2] - x[1]),
            1.5 * (x[1] - x[2]) + 0.5 * (x[3] - x[2]),
            0.5 * (x[2] - x[3]),
        ]
    end
end
