Telemetry Instrumentation Playbook
=================================

Scope & Intent
--------------
Document the near-term plan for instrumenting the specification-to-solution pipeline with OpenTelemetry tracing, Prometheus metrics, and Grafana dashboards. The playbook anchors Action Item #2 from the Master Execution Blueprint and defines the deliverables required to close the instrumentation spike.

Observability Stack
-------------------
* **Tracing & Metrics SDK:** ``opentelemetry-sdk`` with ``opentelemetry-exporter-otlp`` for spans and metrics.
* **Metric Scrape:** ``prometheus_client`` exposition via HTTP endpoint embedded in the orchestrator service (future ``tools/telemetry_gateway.py``).
* **Dashboards:** Grafana provisioned dashboards under ``monitoring/grafana/dashboards/`` with JSON templates checked into source control.
* **Log Correlation:** Python ``structlog`` with OTEL context injection (trace/span IDs) to allow deterministic cross-referencing between structured logs and spans.

Core Signals
------------
``Trace Spans``
    * ``spec.parse`` — boundaries for DSL parsing and schema validation.
    * ``plan.compile_graph`` — orchestration layer translating specs into execution graphs.
    * ``codegen.lower`` — wraps each backend lowering call, tagging backend name, operator, clause ID, and lowering duration.
    * ``compile.execute`` — per-backend compilation including toolchain invocation, exit status, and binary outputs.
    * ``verify.run`` — formal and compliance checks (Frama-C, CBMC, fuzzers, ReviewBot scoring).
    * ``distribute.package`` — packaging, signing, and bundle assembly operations.

``Metrics``
    * ``pipeline_latency_seconds`` (histogram) — total duration from spec ingest to distribution event.
    * ``lowering_duration_seconds`` (histogram) — per backend/operator lowering latency.
    * ``compile_success_total`` / ``compile_failure_total`` (counters) — toolchain outcomes tagged by backend and host.
    * ``verification_block_rate`` (gauge) — ratio of blocked artefacts in current interval.
    * ``offline_bundle_size_bytes`` (summary) — packaged bundle size by target and revision.

``Logs``
    * Structured log events keyed by ``event`` field (``compile.start``, ``compile.end``, ``verify.block``) with trace context to support offline audits.

Instrumentation Targets
-----------------------
Stage 1 (Weeks 0-4)
    * Implement tracing and metric decorators around backend lowering and compile orchestration.
    * Emit Prometheus metrics for registry readiness checks (missing binaries, maturity distribution).
    * Provide lightweight CLI ``tools/telemetry_probe.py`` to run a single backend lowering + compile cycle with instrumentation enabled.

Stage 2 (Weeks 5-12)
    * Extend instrumentation to planner/executor microservices and LLM interactions.
    * Capture dataset/model version metadata as span attributes.

Stage 3 (Weeks 13-20)
    * Instrument verification suite runners, compliance evidence collectors, and ReviewBot feedback loops.
    * Enforce span-linked provenance IDs for compliance artifacts.

Span & Metric Schema
--------------------
* ``trace_id``: 16-byte hex string, propagated via W3C Trace Context.
* ``span attributes``: ``backend.name``, ``operator.name``, ``clause.id`` (hash of clause text), ``toolchain.binary``, ``host.kind`` (online/offline), ``build.cache_hit`` (boolean), ``error.code`` when failures occur.
* ``metric labels``: ``backend``, ``operator``, ``maturity``, ``environment`` (``online``/``offline``), ``host_region``.

Grafana Dashboard Skeleton
--------------------------
Panels to deliver in the spike:
    1. "Pipeline Latency" — heatmap and percentile table for ``pipeline_latency_seconds``.
    2. "Backend Readiness" — stacked bar showing maturity mix and missing binaries count.
    3. "Compile Outcomes" — counter panel combining ``compile_success_total`` and ``compile_failure_total`` with backend filters.
    4. "Verification Blocks" — gauge + log panel surfacing ``verification_block_rate`` and correlated ``verify.block`` events.
    5. "Offline Bundle Size" — line chart tracking ``offline_bundle_size_bytes`` per release.

Implementation Checklist
------------------------
1. Add ``telemetry`` extras to ``pyproject.toml`` (to be created) with pinned OTEL + Prometheus dependencies.
2. Create ``tools/telemetry/__init__.py`` and ``tools/telemetry/span_helpers.py`` providing decorators ``@instrument_span`` and ``record_metric`` utilities.
3. Wire ``BackendDescriptor.render_invocation`` through the instrumentation layer by issuing ``codegen.lower`` spans and updating metrics.
4. Build ``tools/telemetry_probe.py`` CLI to run a representative lowering + compile simulation, emitting sample spans/metrics.
5. Add ``monitoring/grafana/dashboards/pipeline.json`` skeleton with placeholder queries for the metrics defined above.
6. Document environment variables: ``TELEMETRY_EXPORTER_ENDPOINT``, ``PROMETHEUS_PORT``, ``TELEMETRY_ENVIRONMENT``.

Testing Strategy
----------------
* Unit tests: ensure decorators record spans with expected attributes and metrics (use OTEL in-memory exporter).
* Integration tests: run ``telemetry_probe.py`` in CI to confirm metrics endpoint exposes expected series.
* Golden dashboards: validate Grafana JSON against linting tool (``monitoring/scripts/validate_dashboards.py`` to be written).

Open Questions
--------------
* Should the compile fabric push metrics via pull-based Prometheus or adopt OTLP metrics export with agent scraping? (Default to Prometheus for offline parity, reconsider once agent footprint defined.)
* How do we capture toolchain-level stdout/stderr without duplicating log volume? Proposal: attach hashed artifact IDs and store raw logs in object storage referenced by span attributes.
* What is the minimal viable telemetry footprint for air-gapped deployments? Need sizing benchmarks once OTEL exporters are in place.

Next Steps
----------
* Draft ``tools/telemetry/span_helpers.py`` with instrumentation scaffolding (see Action Item 2).
* Define data contracts for span attributes and metric labels in ``docs/data_contracts.rst`` (future work).
* Align with compliance team on evidence retention windows to ensure telemetry logs support audits.

