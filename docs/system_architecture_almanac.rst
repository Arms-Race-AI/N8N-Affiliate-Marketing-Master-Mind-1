System Architecture Almanac
===========================

Purpose
-------
Provide a living map of the specification-to-solution platform so every workstream understands how source artefacts, reasoning services, code generators, verification harnesses, and delivery channels line up. The almanac emphasises current implementation reality, active build-out, and planned end-state so we can harden the foundation without losing sight of the broader flywheel.

Platform Overview
-----------------
``Specification Layer``
    * Status: formative. The long-term goal is a modular ``.upel`` DSL with schema packs per vertical (defense avionics, fintech ledgers, medtech devices, industrial IoT).
    * Current artefacts: symbolic operator catalogue (`upel4/src/upeel/symbolic_operators.py`) representing physical laws and planning primitives. Operators carry clauses that trace code back to governing equations when surfaced downstream.
    * Gaps: parser (`parser/`), scaffolding tools (`tools/scaffolder.py`), and knowledge graph integrations are not yet landed. These remain top priorities in the Weeks 0-4 backlog.

``Reasoning & Planning Layer``
    * Status: planned. Target system orchestrates LLM planners plus deterministic templates to translate specs into execution graphs.
    * Current artefacts: operator registry doubles as the deterministic fallback set; tests (`tests/test_ops_t7.py`) ensure evaluation parity for key operators (gradient, scalar fields, uncertainty relations).
    * Gaps: microservices, prompt libraries, and evaluation harnesses (`registry/`, `orchestrator/`) are pending.

``Multi-Backend Code Generation``
    * Status: partially delivered. Backend registry (`upel4/src/upeel/codegen/backends.py`) and helper utilities (`upel4/src/upeel/codegen/helpers.py`) enumerate 70+ compilation, proof, and hardware targets with metadata for invocation kind, toolchain binaries, and clause-aware lowering.
    * Current artefacts: every operator receives automatically generated ``lower_<backend>`` methods that stitch clause commentary into emitted code, ensuring provenance travels with generated snippets.
    * Gaps: IR annotations (`upel4/src/upeel/ir/`) and template specialisations per backend have not been implemented yet.

``Verification & Compliance``
    * Status: aspirational. ReviewBot policies, formal tooling adapters, and compliance packs remain to be imported (`reviewbot/`, `formal/`, `compliance/`).
    * Current artefacts: clause metadata and toolchain registry provide anchor points for traceability; tests enforce clause presence and comment rendering.
    * Gaps: Bazel integration, evidence capture, Sigstore signing, and compliance data lake interfaces are outstanding items for the Weeks 13-20 programme.

``Distribution & Telemetry``
    * Status: surface-level. The BLACKTOP_SSD artefact is an interactive HTML notebook (`BLACKTOP_SSD/index.html`) seeded with vendor libraries under `BLACKTOP_SSD/vendor/` for rapid demos.
    * Current artefacts: documentation skeleton (``docs/``) with master blueprint and this almanac.
    * Gaps: APIs (`api/`), Web IDE (`webide/`), monitoring stack (`monitoring/`), and GTM dashboards are future milestones.

Execution Flow (As-Is)
----------------------
1. **Operators defined**
       Domain operators are registered in `symbolic_operators.py` with execution semantics, clause metadata, and textual lowering hooks.
2. **Runtime invocation**
       ``import upel4`` exposes operator callables via the dynamic ``__getattr__`` hook (`upel4/src/upeel/__init__.py`).
3. **Symbolic evaluation**
       Operators evaluate against Python data structures (scalars, sequences, matrices). Helper utilities implement the required linear algebra and statistical primitives.
4. **Backend lowering**
       Consumers call ``Op.lower_<backend>()``; the generated snippet embeds clause comments via `BackendDescriptor.render_invocation`, ensuring provenance travels with emitted code.
5. **Testing and validation**
       Pytest suites (`tests/`) verify registry coverage, clause rendering, maturity metadata, and numeric correctness for flagship operators.

Planned End-State Flow
----------------------
``Spec Authoring`` -> ``Parser + Schema Packs`` -> ``Planner/Executor`` -> ``IR Enrichment`` -> ``Backend Emitters`` -> ``Compile Fabric`` -> ``Verification & Compliance`` -> ``Distribution & Telemetry``.

Each stage must deposit signed evidence artefacts, update telemetry, and meet offline parity targets. This almanac will evolve as we land new directories/modules.

Key Interfaces & Handoffs
-------------------------
* **Specification -> Planning**: parser to emit typed IR with provenance tags; planner attaches execution graph metadata. TODO: define `parser/ir_types.py` once schema packs are drafted.
* **Planning -> Codegen**: planner selects operators/backends and passes clause IDs. Existing backend registry is ready to consume these structures.
* **Codegen -> Compile Fabric**: `BACKEND_REGISTRY` enumerates required binaries; compile orchestrator must respect `BackendDescriptor.required_binaries()` and handle sandbox detection.
* **Compile Fabric -> Verification**: generated artefacts should carry clause comments and provenance IDs; verification pipeline attaches signed evidence referencing the same IDs.
* **Verification -> Distribution**: compliance packs gate release packaging; telemetry exporters feed Prometheus/Grafana before GTM launch.

Tooling Landscape
-----------------
* **Python Package**: ``upel4`` is the namespace package. Entry point uses dynamic attribute access to expose operator functions. Future release packaging will publish this as the canonical SDK.
* **Backends Inventory**: 71 descriptors spanning compiled languages (C, Rust, Zig), proofs (Coq, Agda, Isabelle), hardware flows (Verilog, VHDL, Vitis), and data tooling (R, Julia). Maturity ranges from ``stable`` to ``planned``; readiness checks rely on PATH detection via ``shutil.which``.
* **Testing Framework**: Pytest provides coverage; extend with property-based and golden generation tests once IR templates land.
* **Documentation**: Sphinx site anchors around master plan and system guides (currently stubs). This almanac becomes the authoritative architecture reference.

Risk Posture & Controls
-----------------------
* **Model/Planning Drift**: mitigate with clause-based provenance and deterministic fallbacks; schedule quarterly benchmark suites once planner services exist.
* **Toolchain Divergence**: rely on backend readiness checks; planned nightly validation builds via compile fabric.
* **Offline Staleness**: document OCI bundle assembly under ``docs/offline`` (to-do) and ensure clause metadata survives air-gapped signatures.
* **Compliance Gaps**: incorporate clause IDs into ReviewBot prompts and evidence manifests; integrate Sigstore signing flow in future iterations.

Immediate Documentation Hooks
-----------------------------
* Track parser roadmap in a forthcoming ``docs/parser_almanac.rst`` once schema audits complete.
* After telemetry instrumentation spikes, append Grafana dashboard topology under ``docs/telemetry_playbook.rst``.
* Mirror this almanac in offline deliverables to honour parity requirements.

Next Refresh Cadence
--------------------
Update this document at the close of each week in the Weeks 0-4 hardening phase, then align with release train milestones. All major architectural decisions (new IR, planner protocols, compliance flows) must land here before implementation to preserve shared context.


