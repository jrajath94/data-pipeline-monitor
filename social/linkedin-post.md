# LinkedIn Post: data-pipeline-monitor

**I just open-sourced data-pipeline-monitor — here's why it matters.**

Most ML teams discover their production model is broken _after_ it costs them. A customer's experience degrades. Revenue drops. Only then do engineers realize: the input data distribution shifted and nobody was watching.

**The Problem: Monitoring Gap**

The current options are bad:

- Pay $1k+/month for WhyLabs or Evidently (infrastructure overkill for many teams)
- Build in-house monitoring (expensive, time-consuming)
- Deploy with no drift detection (pray it works)

What's missing: lightweight statistical drift detection that integrates directly into your existing monitoring stack (Prometheus/Grafana) and runs in < 5ms.

**My Approach**

I built a library that uses rigorous statistical tests instead of complex ML:

- **Kolmogorov-Smirnov test** for numerical features (catches distribution shifts)
- **Chi-square goodness-of-fit** for categorical data
- **Real p-values** for alerts (no arbitrary thresholds)

Why this works: These tests are proven, interpretable, and fast. I can detect drift on 10K samples in 226ms.

**Technical Highlights**

• 87% test coverage (26 tests, all passing)
• Prometheus integration (works with Grafana, PagerDuty, Datadog)
• Async-ready for FastAPI applications
• Zero external data dependencies

The hardest part: handling categorical drift with unequal batch sizes. Chi-square needs observed and expected counts to match. Solution? Normalize baseline proportions to current sample size. Lesson learned: hidden assumptions in scipy bite hard. That's why thorough testing matters.

**Real Numbers**

- Single feature drift detection: 2.41ms
- 3-feature monitoring: 48ms
- 10K samples + 5 features: 226ms

This fits sub-second SLA requirements in production systems.

**Why I Built This**

At JPMorgan, we had trading models that would quietly underperform for weeks. Post-mortem analysis revealed gradual input distribution shifts we weren't monitoring. I realized: production teams need statistical rigor, not infrastructure complexity.

→ GitHub: github.com/jrajath94/data-pipeline-monitor

**Next Steps**

I'm using this as the foundation for a larger observability toolkit. Days 12-20 of my open-source sprint focus on: distributed training fault tolerance, lightweight feature stores, GPU-accelerated risk engines, and low-latency matching engines for quantitative finance.

If you're building ML systems at scale, drift detection is non-negotiable. I'd love feedback on this approach.

#MachineLearning #MLOps #SoftwareEngineering #OpenSource #Python #Monitoring #DataEngineering
