# TODO

- [x] Spurious Current Laplace jump static is not working well.
- [x] Add two-phase prescribed moving embedded boundary solver and examples
- [ ] The two-phase prescribed moving embedded boundary solver is not working well for the MMS convergence : solver explosion and non convergent for CN
- [ ] Add one-phase free-boundary solver and examples.
- [x] Fix moving monophasic MMS temporal convergence on embedded moving interface (current observed orders are ~0.5-0.73 for BE/CN/θ in `examples/25_moving_mms_time_schemes.jl`).
- [x] Add diphasic moving-interface MMS convergence benchmark for `MovingStokesModelTwoPhase`.
