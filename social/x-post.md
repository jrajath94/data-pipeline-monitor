# 6-Tweet Thread: data-pipeline-monitor

## Tweet 1 (Hook)

ML models fail silently. I built drift detection in 2.4ms.

Most teams find out their model is broken _after_ financial impact. I open-sourced a lightweight monitoring system that catches distribution shifts in production before they destroy accuracy.

→ github.com/jrajath94/data-pipeline-monitor
🧵

## Tweet 2 (The Problem)

The gap in the market:

- WhyLabs, Evidently: $1k+/month, heavy infrastructure
- Nothing: teams just pray

What you need: fast (sub-5ms), statistical (KS + chi-square tests), self-hosted drift detection that plugs into Prometheus.

## Tweet 3 (The Approach)

Instead of complex ML-based drift detection, I use industry-standard statistical hypothesis tests:

• Kolmogorov-Smirnov for numerical features
• Chi-square for categorical
• Real p-values for alerting

Simpler = faster = more interpretable

[Architecture diagram showing: baseline → KS/chi-square test → Prometheus → Grafana]

## Tweet 4 (The Technical Bit)

Hardest problem: categorical drift with unequal batch sizes. Chi-square requires observed+expected counts to match.

Solution: normalize baseline proportions to current sample size. Hidden assumptions in scipy bite hard. That's why testing matters.

## Tweet 5 (The Numbers)

Real benchmarks (M1 MacBook):
• Single feature: 2.41ms
• 3 features: 48ms
• 10K samples + 5 features: 226ms

87% test coverage. 26 tests passing. Prometheus integration. Ready for production.

## Tweet 6 (CTA)

Star it if useful. What should I build next?

→ github.com/jrajath94/data-pipeline-monitor

Built for:
#MachineLearning #MLOps #Monitoring #OpenSource #Python
