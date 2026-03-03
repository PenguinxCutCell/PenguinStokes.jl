# TODO


In 04, the dominant error is at boundary-adjacent momentum rows.

Your own added diagnostics show:

interior truncation (rIntInf) scales about 
h
4
h 
4
 ,
boundary truncation (rBndInf) scales about 
h
1
h 
1
 .
So global velocity error is driven by boundary rows, giving ~first-order behavior.

Why:

04 has nonzero pressure gradient and nontrivial wall Dirichlet velocity.
The discrete pressure-gradient/divergence coupling near walls is not a fully second-order wall closure in this setup.
Interior operators are fine; wall-adjacent consistency is the limiter.
So the issue is not mainly the analytic MMS, and not interior G/H consistency; it is the wall closure for pressure-coupled momentum equations.


Short answer:

It is not mainly pin vs mean gauge.
It is mainly the pressure-gradient/momentum wall closure at boundary-adjacent rows (with Dirichlet velocity walls).
What your results indicate:

PinPressureGauge and MeanPressureGauge both keep ~first-order velocity in 04 (similar trend).
Interior residual is high-order, boundary residual is first-order.
So the limiter is local to wall-adjacent discrete equations, not the global pressure constant mode.
For your MAC face-bounded layout:

Yes, velocity Dirichlet is natural at box faces.
But then momentum rows next to the wall need a wall-consistent pressure term (and fully matched grad/div wall treatment) to keep second order with nonzero pressure.
If that closure is only first-order (or inconsistent with the velocity wall treatment), you get exactly what 04 shows.
So your intuition is right: this is a boundary pressure-gradient consistency issue, not a gauge-choice issue.


