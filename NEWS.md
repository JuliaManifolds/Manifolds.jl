# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0]

### Added

- Vector bundles are generalized to fiber bundles.

### Changed

- Sizes of all manifolds can now be either encoded in type or stored in a field to avoid over-specialization.
  The default is set to store the size in a field. To obtain the old behavior, pass the `parameter=:type` keyword
  argument to manifold constructor. Related changes:
  - `SpecialEuclidean{N}` renamed to `StaticSpecialEuclidean{N}`.
