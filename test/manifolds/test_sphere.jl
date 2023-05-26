using Manifolds, Test
using Manifolds: find_manifold_functions, ManifoldFeatures, ManifoldExpectations
using Manifolds: test_manifold, has_feature_expectations

M = Sphere(2)
f = ManifoldFeatures(;
    functions=find_manifold_functions(M),
    AbstractRetractionMethod[],
    AbstractInverseRetractionMethod[],
    AbstractVectorTransportMethod[],
)
e = ManifoldExpectations(
    Dict(:manifold_dimension => 2),
    Dict(:repr => "Sphere(2, ℝ)"),
    Dict(:exp => 1e-9),
)

ps = [[1.0, 0.0, 0.0], 1 / sqrt(2) .* [1.0, 1.0, 0.0], 1 / sqrt(2) .* [1.0, 0.0, 1.0]]

Xs = [1 / sqrt(2) .* [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

test_manifold(M; points=ps, tangent_vectors=Xs, features=f, expectations=e)
