"""Backend metadata and orchestration helpers for lowering targets."""
from __future__ import annotations

from dataclasses import dataclass
from shutil import which
from typing import Dict, Iterable, Sequence

from .helpers import COMMENT_SYNTAX, comment


@dataclass(frozen=True)
class BackendDescriptor:
    """Describe how to emit and compile code for a backend target."""

    name: str
    language: str
    file_extension: str
    invocation_kind: str
    statement_terminator: str = ";"
    compile_cmd: tuple[str, ...] | None = None
    run_cmd: tuple[str, ...] | None = None
    maturity: str = "planned"
    notes: str = ""
    extra_requires: tuple[str, ...] = ()

    def required_binaries(self) -> tuple[str, ...]:
        """Return the toolchain binaries that should exist on PATH."""
        seen: list[str] = []

        def _record(candidate: str) -> None:
            if not candidate:
                return
            if "{" in candidate or "}" in candidate:
                return
            if candidate not in seen:
                seen.append(candidate)

        if self.compile_cmd:
            _record(self.compile_cmd[0])
        if self.run_cmd:
            _record(self.run_cmd[0])
        for extra in self.extra_requires:
            _record(extra)
        return tuple(seen)

    def missing_binaries(self) -> tuple[str, ...]:
        """List missing toolchain commands."""
        return tuple(cmd for cmd in self.required_binaries() if which(cmd) is None)

    def is_ready(self) -> bool:
        """Return True if every required binary appears to be installed."""
        return not self.missing_binaries()

    def render_invocation(self, op_name: str, args: Sequence[str], clause: str, indent: int = 0) -> str:
        """Render a backend flavored invocation with a clause aware comment."""
        rendered_args = ", ".join(args)
        call = f"{op_name}({rendered_args})" if rendered_args else f"{op_name}()"
        if self.statement_terminator:
            call = f"{call}{self.statement_terminator}"
        clause_comment = comment(self.name, f"{op_name}: {clause}")
        if indent:
            pad = " " * indent
            return f"{pad}{clause_comment}\n{pad}{call}"
        return f"{clause_comment} {call}"


def _build_backend_registry() -> Dict[str, BackendDescriptor]:
    descriptors = [
        BackendDescriptor(
            name="ada",
            language="Ada",
            file_extension=".adb",
            invocation_kind="compiled",
            compile_cmd=("gnatmake", "{source}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Leverages GNAT to build Ada executables; adjust {output} when GNAT chooses a different name.",
        ),
        BackendDescriptor(
            name="agda",
            language="Agda",
            file_extension=".agda",
            invocation_kind="proof",
            statement_terminator="",
            compile_cmd=("agda", "{source}"),
            maturity="alpha",
            notes="Invokes the Agda compiler and type checker.",
        ),
        BackendDescriptor(
            name="armasm",
            language="ARM Assembly",
            file_extension=".s",
            invocation_kind="assembly",
            compile_cmd=("arm-none-eabi-as", "{source}", "-o", "{output}.o"),
            run_cmd=("arm-none-eabi-objdump", "-d", "{output}.o"),
            maturity="alpha",
            notes="Assembles with the GNU Arm Embedded toolchain; linking is left to the caller.",
        ),
        BackendDescriptor(
            name="bqn",
            language="BQN",
            file_extension=".bqn",
            invocation_kind="interpreted",
            statement_terminator="",
            run_cmd=("bqn", "{source}"),
            maturity="beta",
            notes="Requires the BQN reference interpreter.",
        ),
        BackendDescriptor(
            name="bsv",
            language="Bluespec SystemVerilog",
            file_extension=".bsv",
            invocation_kind="hdl",
            compile_cmd=("bsc", "{source}"),
            maturity="alpha",
            notes="Uses the Bluespec compiler; downstream elaboration flow is user defined.",
        ),
        BackendDescriptor(
            name="c",
            language="C",
            file_extension=".c",
            invocation_kind="compiled",
            compile_cmd=("gcc", "{source}", "-std=c17", "-O2", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="stable",
            notes="Defaults to GCC; swap for clang or another compiler as needed.",
        ),
        BackendDescriptor(
            name="chapel",
            language="Chapel",
            file_extension=".chpl",
            invocation_kind="compiled",
            compile_cmd=("chpl", "{source}", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Assumes the Chapel compiler is installed.",
        ),
        BackendDescriptor(
            name="chisel",
            language="Chisel (Scala)",
            file_extension=".scala",
            invocation_kind="hdl",
            statement_terminator="",
            compile_cmd=("sbt", "run"),
            maturity="alpha",
            notes="Expects an SBT project with a Chisel main; command runs the default target.",
        ),
        BackendDescriptor(
            name="coq",
            language="Coq",
            file_extension=".v",
            invocation_kind="proof",
            statement_terminator="",
            compile_cmd=("coqc", "{source}"),
            maturity="alpha",
            notes="Compiles Coq developments with coqc.",
        ),
        BackendDescriptor(
            name="cpp",
            language="C++",
            file_extension=".cc",
            invocation_kind="compiled",
            compile_cmd=("g++", "{source}", "-std=c++20", "-O2", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="stable",
            notes="Uses g++; adjust flags to match local standards.",
        ),
        BackendDescriptor(
            name="crystal",
            language="Crystal",
            file_extension=".cr",
            invocation_kind="compiled",
            compile_cmd=("crystal", "build", "{source}", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Relies on the Crystal compiler.",
        ),
        BackendDescriptor(
            name="cs",
            language="C#",
            file_extension=".cs",
            invocation_kind="compiled",
            compile_cmd=("csc", "{source}", "/out:{output}.exe"),
            run_cmd=("{output}.exe",),
            maturity="beta",
            notes="Targets the Roslyn csc compiler; use dotnet CLI for project builds.",
        ),
        BackendDescriptor(
            name="cuda",
            language="CUDA C++",
            file_extension=".cu",
            invocation_kind="kernel",
            compile_cmd=("nvcc", "{source}", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Requires NVIDIA's CUDA toolkit and nvcc.",
        ),
        BackendDescriptor(
            name="d",
            language="D",
            file_extension=".d",
            invocation_kind="compiled",
            compile_cmd=("dmd", "{source}", "-of{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Uses the DMD compiler; interchange with ldc2 where appropriate.",
        ),
        BackendDescriptor(
            name="dart",
            language="Dart",
            file_extension=".dart",
            invocation_kind="compiled",
            compile_cmd=("dart", "compile", "exe", "{source}", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Builds native Dart executables via the Dart SDK.",
        ),
        BackendDescriptor(
            name="fortran",
            language="Fortran",
            file_extension=".f90",
            invocation_kind="compiled",
            compile_cmd=("gfortran", "{source}", "-O2", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Assumes GNU Fortran; tweak flags for other compilers.",
        ),
        BackendDescriptor(
            name="futhark",
            language="Futhark",
            file_extension=".fut",
            invocation_kind="compiled",
            compile_cmd=("futhark", "opencl", "{source}"),
            maturity="alpha",
            notes="Compiles to an OpenCL host; generated artifacts live next to the source file.",
        ),
        BackendDescriptor(
            name="go",
            language="Go",
            file_extension=".go",
            invocation_kind="compiled",
            statement_terminator="",
            compile_cmd=("go", "build", "-o", "{output}", "{source}"),
            run_cmd=("{output}",),
            maturity="stable",
            notes="Uses the Go toolchain; semicolons are implicit in snippets.",
        ),
        BackendDescriptor(
            name="hs",
            language="Haskell",
            file_extension=".hs",
            invocation_kind="compiled",
            statement_terminator="",
            compile_cmd=("ghc", "{source}", "-O2", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Relies on GHC for native code generation.",
        ),
        BackendDescriptor(
            name="intel_hls",
            language="Intel HLS",
            file_extension=".cpp",
            invocation_kind="hls",
            compile_cmd=("i++", "{source}", "-o", "{output}.out"),
            maturity="alpha",
            notes="Invokes the Intel HLS i++ compiler; synthesis steps remain manual.",
        ),
        BackendDescriptor(
            name="isabelle",
            language="Isabelle",
            file_extension=".thy",
            invocation_kind="proof",
            statement_terminator="",
            compile_cmd=("isabelle", "build", "{session}"),
            run_cmd=("isabelle", "tty", "{session}"),
            maturity="alpha",
            notes="Uses the Isabelle build tool; {session} should match your theory session.",
        ),
        BackendDescriptor(
            name="java",
            language="Java",
            file_extension=".java",
            invocation_kind="compiled",
            compile_cmd=("javac", "{source}"),
            run_cmd=("java", "{main_class}"),
            maturity="stable",
            notes="Compiles standalone Java sources; configure {main_class} for execution.",
        ),
        BackendDescriptor(
            name="js",
            language="JavaScript",
            file_extension=".js",
            invocation_kind="interpreted",
            run_cmd=("node", "{source}"),
            maturity="stable",
            notes="Executes JavaScript via Node.js.",
        ),
        BackendDescriptor(
            name="julia",
            language="Julia",
            file_extension=".jl",
            invocation_kind="interpreted",
            statement_terminator="",
            run_cmd=("julia", "{source}"),
            maturity="stable",
            notes="Runs Julia scripts with the julia binary.",
        ),
        BackendDescriptor(
            name="kotlin",
            language="Kotlin",
            file_extension=".kt",
            invocation_kind="compiled",
            compile_cmd=("kotlinc", "{source}", "-include-runtime", "-d", "{output}.jar"),
            run_cmd=("java", "-jar", "{output}.jar"),
            maturity="beta",
            notes="Produces a runnable jar via kotlinc.",
        ),
        BackendDescriptor(
            name="lean",
            language="Lean",
            file_extension=".lean",
            invocation_kind="proof",
            statement_terminator="",
            compile_cmd=("lean", "{source}"),
            maturity="alpha",
            notes="Checks Lean developments with the lean binary.",
        ),
        BackendDescriptor(
            name="lua",
            language="Lua",
            file_extension=".lua",
            invocation_kind="interpreted",
            statement_terminator="",
            run_cmd=("lua", "{source}"),
            maturity="stable",
            notes="Uses the Lua interpreter.",
        ),
        BackendDescriptor(
            name="m",
            language="MATLAB/Octave",
            file_extension=".m",
            invocation_kind="interpreted",
            statement_terminator="",
            run_cmd=("octave", "--quiet", "--eval", "run('{source}')"),
            maturity="beta",
            notes="Runs scripts via Octave; replace with MATLAB -batch if available.",
        ),
        BackendDescriptor(
            name="metal",
            language="Metal",
            file_extension=".metal",
            invocation_kind="kernel",
            compile_cmd=("metal", "{source}", "-o", "{output}.air"),
            run_cmd=("metallib", "{output}.air", "-o", "{output}.metallib"),
            maturity="alpha",
            notes="Two step compile: metal emits AIR, metallib packages a library.",
        ),
        BackendDescriptor(
            name="mikroc",
            language="MikroC",
            file_extension=".c",
            invocation_kind="embedded",
            compile_cmd=("mikroc", "{source}"),
            maturity="alpha",
            notes="CLI entry for MikroC; device specific flags supplied externally.",
        ),
        BackendDescriptor(
            name="misrac",
            language="MISRA C",
            file_extension=".c",
            invocation_kind="compiled",
            compile_cmd=("gcc", "{source}", "-std=c90", "-Wall", "-Wextra", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Uses GCC with strict C90 flags; integrate dedicated MISRA analyzers separately.",
        ),
        BackendDescriptor(
            name="nim",
            language="Nim",
            file_extension=".nim",
            invocation_kind="compiled",
            statement_terminator="",
            compile_cmd=("nim", "c", "-d:release", "{source}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Nim names binaries after the module; adjust {output} accordingly when invoking.",
        ),
        BackendDescriptor(
            name="ocaml",
            language="OCaml",
            file_extension=".ml",
            invocation_kind="compiled",
            statement_terminator="",
            compile_cmd=("ocamlopt", "{source}", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Uses ocamlopt for native code.",
        ),
        BackendDescriptor(
            name="omp_f",
            language="OpenMP Fortran",
            file_extension=".f90",
            invocation_kind="compiled",
            compile_cmd=("gfortran", "-fopenmp", "{source}", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Compiles OpenMP workloads with gfortran.",
        ),
        BackendDescriptor(
            name="opencl",
            language="OpenCL C",
            file_extension=".cl",
            invocation_kind="kernel",
            compile_cmd=("clang", "-cl-std=CL2.0", "{source}", "-o", "{output}.bc"),
            maturity="alpha",
            notes="Uses clang to emit SPIR bitcode; many runtimes JIT kernels at execution time.",
        ),
        BackendDescriptor(
            name="r",
            language="R",
            file_extension=".R",
            invocation_kind="interpreted",
            statement_terminator="",
            run_cmd=("Rscript", "{source}"),
            maturity="stable",
            notes="Executes R scripts via Rscript.",
        ),
        BackendDescriptor(
            name="rb",
            language="Ruby",
            file_extension=".rb",
            invocation_kind="interpreted",
            statement_terminator="",
            run_cmd=("ruby", "{source}"),
            maturity="stable",
            notes="Runs Ruby scripts with the reference interpreter.",
        ),
        BackendDescriptor(
            name="rust",
            language="Rust",
            file_extension=".rs",
            invocation_kind="compiled",
            compile_cmd=("rustc", "{source}", "-O", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="stable",
            notes="Compiles with rustc using optimized settings.",
        ),
        BackendDescriptor(
            name="slx",
            language="Simulink",
            file_extension=".slx",
            invocation_kind="model",
            statement_terminator="",
            compile_cmd=("matlab", "-batch", "slbuild('{model}')"),
            maturity="alpha",
            notes="Invokes MATLAB slbuild; {model} should match the Simulink system.",
        ),
        BackendDescriptor(
            name="spinal",
            language="SpinalHDL (Scala)",
            file_extension=".scala",
            invocation_kind="hdl",
            statement_terminator="",
            compile_cmd=("sbt", "run"),
            maturity="alpha",
            notes="Assumes an SBT SpinalHDL project; generation flow is user managed.",
        ),
        BackendDescriptor(
            name="spirv",
            language="SPIR-V",
            file_extension=".spvasm",
            invocation_kind="intermediate",
            statement_terminator="",
            compile_cmd=("spirv-as", "{source}", "-o", "{output}.spv"),
            run_cmd=("spirv-dis", "{output}.spv"),
            maturity="alpha",
            notes="Assembles and disassembles SPIR-V modules.",
        ),
        BackendDescriptor(
            name="st",
            language="Structured Text",
            file_extension=".st",
            invocation_kind="plc",
            compile_cmd=("codesyscmd", "compile", "{project}"),
            maturity="alpha",
            notes="Placeholder for IEC 61131-3 toolchains; supply project level automation.",
        ),
        BackendDescriptor(
            name="sv",
            language="SystemVerilog",
            file_extension=".sv",
            invocation_kind="hdl",
            compile_cmd=("iverilog", "-g2012", "-o", "{output}", "{source}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Uses Icarus Verilog with SystemVerilog support enabled.",
        ),
        BackendDescriptor(
            name="swift",
            language="Swift",
            file_extension=".swift",
            invocation_kind="compiled",
            compile_cmd=("swiftc", "{source}", "-O", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="stable",
            notes="Compiles Swift sources with swiftc.",
        ),
        BackendDescriptor(
            name="sycl",
            language="SYCL",
            file_extension=".cpp",
            invocation_kind="kernel",
            compile_cmd=("dpcpp", "{source}", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="alpha",
            notes="Uses Intel's DPC++ compiler for SYCL workloads.",
        ),
        BackendDescriptor(
            name="ts",
            language="TypeScript",
            file_extension=".ts",
            invocation_kind="transpiled",
            compile_cmd=("tsc", "{source}"),
            run_cmd=("node", "{output}"),
            maturity="beta",
            notes="Transpiles with tsc; {output} should reference the emitted JavaScript bundle.",
        ),
        BackendDescriptor(
            name="verilog",
            language="Verilog",
            file_extension=".v",
            invocation_kind="hdl",
            compile_cmd=("iverilog", "-o", "{output}", "{source}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Simulates Verilog designs with Icarus Verilog.",
        ),
        BackendDescriptor(
            name="vhdl",
            language="VHDL",
            file_extension=".vhd",
            invocation_kind="hdl",
            compile_cmd=("ghdl", "-a", "{source}"),
            run_cmd=("ghdl", "-r", "{entity}"),
            maturity="beta",
            notes="Analyzes with GHDL; {entity} selects the design unit to execute.",
        ),
        BackendDescriptor(
            name="vitis",
            language="Vitis HLS",
            file_extension=".cpp",
            invocation_kind="hls",
            compile_cmd=("v++", "-c", "{source}", "-o", "{output}.xo"),
            maturity="alpha",
            notes="Generates XO kernels using Xilinx v++.",
        ),
        BackendDescriptor(
            name="wasm",
            language="WebAssembly (wat)",
            file_extension=".wat",
            invocation_kind="intermediate",
            statement_terminator="",
            compile_cmd=("wat2wasm", "{source}", "-o", "{output}.wasm"),
            run_cmd=("wasmtime", "{output}.wasm"),
            maturity="beta",
            notes="Assembles text format Wasm and executes with wasmtime.",
        ),
        BackendDescriptor(
            name="zig",
            language="Zig",
            file_extension=".zig",
            invocation_kind="compiled",
            compile_cmd=("zig", "build-exe", "{source}", "-OReleaseSafe", "-o", "{output}"),
            run_cmd=("{output}",),
            maturity="beta",
            notes="Builds Zig executables with release-safe optimizations.",
        ),
    ]
    registry: Dict[str, BackendDescriptor] = {desc.name: desc for desc in descriptors}
    return registry


def _ensure_comment_backends(registry: Dict[str, BackendDescriptor]) -> None:
    for backend in COMMENT_SYNTAX:
        if backend not in registry:
            registry[backend] = BackendDescriptor(
                name=backend,
                language=backend,
                file_extension="",
                invocation_kind="planned",
                statement_terminator="",
                maturity="planned",
                notes="Auto-generated placeholder; supply full toolchain metadata.",
            )


BACKEND_REGISTRY = _build_backend_registry()
_ensure_comment_backends(BACKEND_REGISTRY)


def get_backend(name: str) -> BackendDescriptor:
    try:
        return BACKEND_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown backend '{name}'") from exc


def iter_backends() -> Iterable[BackendDescriptor]:
    for backend_name in sorted(BACKEND_REGISTRY):
        yield BACKEND_REGISTRY[backend_name]


def ready_backends() -> list[BackendDescriptor]:
    return [desc for desc in iter_backends() if desc.is_ready()]


def backend_summary() -> list[dict[str, object]]:
    return [
        {
            "name": desc.name,
            "language": desc.language,
            "kind": desc.invocation_kind,
            "maturity": desc.maturity,
            "ready": desc.is_ready(),
            "missing": desc.missing_binaries(),
        }
        for desc in iter_backends()
    ]


__all__ = [
    "BackendDescriptor",
    "BACKEND_REGISTRY",
    "get_backend",
    "iter_backends",
    "ready_backends",
    "backend_summary",
]