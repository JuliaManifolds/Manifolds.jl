# A Test suite

The submodule `Manifolds.Test` provides functions to test mathematical properties
of functions defined for a manifold according to the API from `ManifoldsBase.jl`.

```@docs
Manifolds.Test
```

The main function is

```@docs
Manifolds.Test.test_manifold
```

## Functions for individual tests

```@docs
Manifolds.Test.test_copy
Manifolds.Test.test_copyto
Manifolds.Test.test_default_retraction
Manifolds.Test.test_default_inverse_retraction
Manifolds.Test.test_default_vector_transport_method
Manifolds.Test.test_distance
Manifolds.Test.test_exp
Manifolds.Test.test_embed
Manifolds.Test.test_embed_project
Manifolds.Test.test_flat
Manifolds.Test.test_injectivity_radius
Manifolds.Test.test_inner
Manifolds.Test.test_inverse_retract
Manifolds.Test.test_is_default_metric
Manifolds.Test.test_is_flat
Manifolds.Test.test_is_point
Manifolds.Test.test_is_vector
Manifolds.Test.test_geodesic
Manifolds.Test.test_get_basis
Manifolds.Test.test_get_coordinates
Manifolds.Test.test_get_embedding
Manifolds.Test.test_get_vector
Manifolds.Test.test_get_vectors
Manifolds.Test.test_log
Manifolds.Test.test_manifold_dimension
Manifolds.Test.test_manifold_volume
Manifolds.Test.test_mid_point
Manifolds.Test.test_norm
Manifolds.Test.test_parallel_transport
Manifolds.Test.test_project
Manifolds.Test.test_rand
Manifolds.Test.test_repr
Manifolds.Test.test_representation_size
Manifolds.Test.test_retract
Manifolds.Test.test_sectional_curvature
Manifolds.Test.test_sharp
Manifolds.Test.test_shortest_geodesic
Manifolds.Test.test_vector_transport
Manifolds.Test.test_volume_density
Manifolds.Test.test_Weingarten
Manifolds.Test.test_zero_vector
```

## Internals to handle expectations

```@docs
Manifolds.Test.AbstractExpectation
Manifolds.Test.Expect
Manifolds.Test.NoExpectation
Manifolds.Test.expect
Manifolds.Test.isexpected
```

## Former tests

```@docs
Manifolds.find_eps
Manifolds.test_manifold
Manifolds.test_parallel_transport
```