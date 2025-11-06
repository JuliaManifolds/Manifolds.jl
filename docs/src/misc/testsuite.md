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
Manifolds.Test.test_norm
Manifolds.Test.test_exp
Manifolds.Test.test_manifold_dimension
Manifolds.Test.test_log
Manifolds.Test.test_inner
```

## Former tests

```@docs
Manifolds.find_eps
Manifolds.test_manifold
Manifolds.test_parallel_transport
```