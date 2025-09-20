Master Execution Blueprint
==========================

Mission
-------
Deliver a single specification-to-solution pipeline that can materialise production-ready systems across all 51+ language, hardware, and proof backends whether the platform is operating online in a hyperscale code farm or offline on an isolated laptop. Every release must ship with deterministic provenance, formal verification artefacts, and enterprise/military-grade operational controls.

Guiding Principles
------------------
* **Maximal Coverage:** Every `.upel` specification can target any backend listed in ``scripts/targets.json`` without manual intervention.
* **Deterministic Trust:** Toolchains, artefacts, and review steps are cryptographically signed and reproducible via ``compile-everywhere.sh`` and ``tools/trace_sign.py``.
* **Continuous Verification:** Formal tooling (Frama-C, JasperGold, proof adapters) gates release; ReviewBot remains the brutal last line of defence.
* **Offline Parity:** Offline deployments receive the same generation, verification, and telemetry capabilities through containerised bundles and "compile farm in a box" kits.
* **Operational Insight:** Telemetry streams into the Prometheus/Grafana stack defined under ``monitoring/`` and surfaces in GTM/customer dashboards.

Target Capability Stack
-----------------------
* **Specification Experience** – Extend the ``parser/`` DSL and ``tools/scaffolder.py`` wizards with vertical schema packs, compliance hints, and embedded risk scoring.
* **Reasoning & Planning Layer** – Introduce an orchestration service that translates specs into execution graphs, selects appropriate LLMs/backends, and coordinates fallbacks to deterministic templates in ``upel4/codegen``.
* **Multi-Backend Code Generation** – Leverage and expand the adapters in ``upel4/codegen`` and the registry under ``registry/`` to automatically emit artefacts for all supported targets (compiled binaries, HDL, proofs, SDKs).
* **Compile Fabric** – Harden ``compile-everywhere.sh`` and ``tools/perf_bench.py`` to auto-benchmark new backends, manage cache invalidation, and orchestrate distributed builders described in ``docker-stack.yaml`` and ``terraform/`` modules.
* **Verification & Compliance** – Chain the verification runners in ``tools/`` and ReviewBot policies (``reviewbot/``) to capture signed evidence, map to compliance packs, and block non-conformant outputs.
* **Distribution & APIs** – Expand ``api/server.py`` and the Helm chart under ``helm/`` for customer onboarding, licensing, and billing while maintaining audit logs and provenance manifests.
* **Telemetry & GTM** – Use ``monitoring/`` configuration, ``tools/metrics_exporter.py``, and ``tools/release_bot.py`` to supply product, sales, and customer success teams with real-time insights.

Execution Roadmap
-----------------
``Weeks 0-4 – Foundation Hardening``
    * Document current architecture into a "System Architecture Almanac" referencing IR flows, Bazel rules (``bazel/rules_upeel.bzl``), and backend lifecycles.
    * Normalise telemetry by instrumenting compile stages with OpenTelemetry spans and Prometheus exporters; publish dashboards via Grafana manifests.
    * Refresh ``perf_cache.json`` through benchmark runs and align deterministic build strategies across online/offline environments.

``Weeks 5-12 – Orchestration & Offline Enablement``
    * Implement the specification wizard upgrades and DSL schema packs for high-priority verticals (defense avionics, fintech ledgers, medtech devices).
    * Ship the LLM orchestration layer with planner/executor services, fallback templating, and sandboxed tool access; integrate with ReviewBot for feedback-driven refinement.
    * Package quantised/distilled model suites and runtime dependencies as signed OCI images, together with Terraform/Docker scripts for air-gapped deployments.

``Weeks 13-20 – Verification & Compliance Expansion``
    * Integrate Frama-C, CBMC, JasperGold, fuzzers, and proof assistant hooks into Bazel targets with automated evidence capture into a compliance data lake.
    * Extend ReviewBot scoring with reinforcement loops that feed critiques back into prompt templates and backlog prioritisation.
    * Establish quality scorecards covering spec completeness, verification coverage, performance benchmarks, and deterministic build metrics.

``Weeks 21-28 – Enterprise Platform & Marketplace``
    * Launch customer-facing APIs, portal, and billing automation built on ``api/``, ``webide/``, and licensing flows in ``tools/license_gen.py``.
    * Curate industry starter kits with specs, generated artefacts, verification configs, and deployment scripts; open marketplace contributions with vetting workflows.
    * Embed telemetry into customer dashboards, enabling evidence downloads and compliance reporting.

``Weeks 29+ – Flywheel & Continuous Improvement``
    * Establish marketing and partner ecosystems leveraging telemetry-driven benchmark reports and field demos.
    * Maintain a voice-of-customer loop that feeds the model prompt library, template updates, and compliance pack refreshes.
    * Continue model distillation, prompt tuning, and expansion of backend support as new languages or toolchains emerge.

Workstreams & Deliverables
--------------------------
Specification & Knowledge
    * Extend ``parser/grammar.upel`` and ``parser/parser.py`` with modular schema packs.
    * Expand ``tools/scaffolder.py`` to auto-generate specs, golden test data, and compliance checklists.
    * Maintain a knowledge graph connecting requirements, controls, and code templates for deterministic planning.

Model & Reasoning
    * Build planner/executor microservices that sequence generation, simulation, review, and verification steps.
    * Curate prompt libraries per industry, including deterministic templates stored under ``registry/``.
    * Implement evaluation harnesses to score LLM outputs versus template baselines before they enter the compile pipeline.

Multi-Backend Pipeline
    * Audit adapters in ``upel4/codegen`` and extend to any missing targets listed in ``scripts/targets.json``.
    * Add semantic annotations to IR nodes (``upel4/src/upeel/ir/nodes.py``) so downstream emitters can choose idiomatic constructs per language.
    * Enhance ``compile-everywhere.sh`` for distributed execution, caching policies, and failure remediation.

Verification & Compliance
    * Codify compliance packs (NIST, SOC2, DO-178C, ISO 26262) into automated checks mapped to verification outputs.
    * Enforce signed provenance artefacts through ``tools/trace_sign.py`` and integrate with Sigstore.
    * Extend ReviewBot prompts (``reviewbot/prompt.txt``) to capture compliance rationale and corrective actions.

Offline & Edge
    * Publish "compile farm in a box" bundles combining Terraform modules, Docker stacks, and curated model artefacts.
    * Provide incremental sync tooling for reconnect scenarios, ensuring provenance chains remain intact.
    * Deliver air-gapped documentation and runbooks, including offline Sphinx builds and incident response guides.

Platform, Distribution & GTM
    * Harden ``api/server.py`` with rate limiting, audit logging, and licensing enforcement.
    * Enhance the Web IDE (``webide/server.py``) with wizard-driven authoring, real-time telemetry, and offline-sync awareness.
    * Develop GTM collateral, pricing models, and partner enablement kits aligned with telemetry insights.

Telemetry & Analytics
    * Instrument builds and deployments with metrics exported via ``tools/metrics_exporter.py`` to Prometheus.
    * Create Grafana dashboards for engineering, compliance, GTM, and customer success stakeholders.
    * Feed anonymised telemetry into marketing automation for benchmark publications and customer storytelling.

Risk & Mitigation
-----------------
* **Model Drift:** Schedule quarterly benchmark runs and update prompt templates alongside retraining/distillation to preserve accuracy and compliance.
* **Toolchain Divergence:** Maintain version-locked Docker images and nightly validation builds to catch compiler or dependency regressions early.
* **Offline Staleness:** Provide signed update bundles and integrity checks so disconnected deployments can upgrade safely.
* **Compliance Gaps:** Automate evidence collection and map to controls; escalate to manual review when automated checks flag deficiencies.

Metrics & Success Criteria
--------------------------
* Generation success rate ≥ 98% across top 20 customer blueprints.
* Verification coverage ≥ 95% for mandatory targets; zero tolerance for unreviewed alarms.
* Deterministic rebuild delta ≤ 1% between online and offline environments.
* Customer onboarding time < 2 days with licence provisioning and telemetry dashboards live.
* Marketplace contribution acceptance rate ≥ 80% with automated vetting.

Immediate Next Actions
----------------------
1. DONE (2025-09-20): Publish the System Architecture Almanac capturing IR flows, toolchains, and handoffs.
   - Location: docs/system_architecture_almanac.rst; refresh weekly through the Weeks 0-4 hardening phase.
2. Instrument compile stages with OpenTelemetry/Prometheus probes and validate dashboards locally.
   - Deliverables: instrumentation playbook, prototype spans around backend lowering + compile loops, Grafana dashboard skeleton.
3. Complete DSL gap analysis for defense, fintech, medtech, and industrial IoT; rank missing constructs.
   - Deliverables: schema pack backlog, parser extension plan, risk scoring hooks.
4. Inventory model assets, licences, and hardware compatibility; outline quantisation/distillation roadmap.
   - Deliverables: asset registry, licence checks, hardware coverage matrix, quantisation sequencing notes.
5. Define compliance baseline requirements and integrate them into ReviewBot scoring criteria.
   - Deliverables: baseline control list, ReviewBot scoring updates, evidence capture workflow.
6. Outline GTM launch plan covering pricing, packaging, and partner recruitment for the first three verticals.
   - Deliverables: pricing guardrails, tiered packaging, partner enablement plan aligned with telemetry insights.

