{
  description = "Natsume Japanese collocation service";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    git-hooks-nix = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.git-hooks-nix.flakeModule ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      perSystem =
        {
          config,
          pkgs,
          system,
          lib,
          ...
        }:
        let
          revision = inputs.self.shortRev or inputs.self.dirtyShortRev or "dirty";
          workspace = inputs.uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
          pythonBase = pkgs.callPackage inputs.pyproject-nix.build.packages {
            python = pkgs.python312;
          };
          mkPythonSet =
            dependencies:
            pythonBase.overrideScope (
              lib.composeManyExtensions [
                inputs.pyproject-build-systems.overlays.wheel
                (workspace.mkPyprojectOverlay {
                  inherit dependencies;
                  sourcePreference = "wheel";
                })
                (
                  final: prev:
                  lib.genAttrs
                    [
                      "docopt"
                      "mosestokenizer"
                      "toolwrapper"
                      "uctools"
                    ]
                    (
                      name:
                      prev.${name}.overrideAttrs (old: {
                        # These legacy sdists execute setuptools but do not
                        # declare a PEP 517 build system.
                        nativeBuildInputs =
                          (old.nativeBuildInputs or [ ])
                          ++ final.resolveBuildSystem {
                            setuptools = [ ];
                          };
                      })
                    )
                )
              ]
            );

          # Selecting CPU here resolves the CPU/CUDA lock conflict. Each virtual
          # environment still contains only the extras named in its own spec.
          cpu-resolution.natsume-simple = [ "cpu" ];
          server-dependencies.natsume-simple = [ "backend" ];
          builder-dependencies.natsume-simple = [
            "builder"
            "cpu"
          ];
          test-dependencies.natsume-simple = [
            "backend"
            "builder"
            "cpu"
            "test"
          ];
          cpuPythonSet = mkPythonSet cpu-resolution;
          serverPython = cpuPythonSet.mkVirtualEnv "natsume-server-python" server-dependencies;
          builderPython = cpuPythonSet.mkVirtualEnv "natsume-builder-python" builder-dependencies;
          testPython = cpuPythonSet.mkVirtualEnv "natsume-test-python" test-dependencies;

          frontend = pkgs.buildNpmPackage {
            pname = "natsume-frontend";
            version = "0.3.0";
            src = ./natsume-frontend;
            npmDepsHash = "sha256-EzpXV1g9dpjpwSBoqKMZPWB4LUg5/HDuf2uurWMrE1s=";
            npmBuildScript = "build";
            installPhase = ''
              runHook preInstall
              mkdir -p "$out"
              cp -r build/. "$out/"
              runHook postInstall
            '';
          };

          server = pkgs.writeShellApplication {
            name = "natsume-serve";
            runtimeInputs = [ serverPython ];
            text = ''
              artifact_dir="''${NATSUME_ARTIFACT_DIR:-/var/lib/natsume/artifact}"
              host="''${NATSUME_HOST:-127.0.0.1}"
              port="''${NATSUME_PORT:-8000}"

              while (( $# )); do
                case "$1" in
                  --artifact-dir)
                    artifact_dir="$2"
                    shift 2
                    ;;
                  --host)
                    host="$2"
                    shift 2
                    ;;
                  --port)
                    port="$2"
                    shift 2
                    ;;
                  -h|--help)
                    echo "usage: natsume-serve [--artifact-dir PATH] [--host HOST] [--port PORT]"
                    exit 0
                    ;;
                  *)
                    echo "natsume-serve: unknown argument: $1" >&2
                    exit 2
                    ;;
                esac
              done

              export NATSUME_ARTIFACT_DIR="$artifact_dir"
              export NATSUME_FRONTEND_DIR=${frontend}
              exec uvicorn natsume_simple.api:app --host "$host" --port "$port"
            '';
          };

          corpusBuilder = pkgs.writeShellApplication {
            name = "natsume-corpus";
            runtimeInputs = [
              builderPython
              pkgs.nkf
              pkgs.pandoc
            ];
            text = ''
              export NATSUME_BUILDER_REVISION=${lib.escapeShellArg revision}
              exec ${builderPython}/bin/natsume-corpus "$@"
            '';
          };

          frontendCheck = pkgs.buildNpmPackage {
            pname = "natsume-frontend-check";
            version = "0.3.0";
            src = ./natsume-frontend;
            npmDepsHash = "sha256-EzpXV1g9dpjpwSBoqKMZPWB4LUg5/HDuf2uurWMrE1s=";
            dontNpmBuild = true;
            doCheck = true;
            checkPhase = ''
              npm run lint
              npm run check
              npm run test:unit -- --run
              npm run build
            '';
            installPhase = ''touch "$out"'';
          };

          backendCheck = pkgs.runCommand "natsume-backend-check" { nativeBuildInputs = [ testPython ]; } ''
            cp -r ${./.} source
            chmod -R u+w source
            cd source
            pytest -m "not nlp_model"
            touch "$out"
          '';

          modelIntegration =
            pkgs.runCommand "natsume-nlp-model-integration" { nativeBuildInputs = [ testPython ]; }
              ''
                cp -r ${./.} source
                chmod -R u+w source
                cd source
                pytest -m nlp_model
                touch "$out"
              '';

          playwrightBrowsers = pkgs.playwright-driver.browsers.override {
            withFirefox = false;
            withWebkit = false;
          };
          playwrightCheck = pkgs.buildNpmPackage {
            pname = "natsume-playwright-check";
            version = "0.3.0";
            src = ./natsume-frontend;
            npmDepsHash = "sha256-EzpXV1g9dpjpwSBoqKMZPWB4LUg5/HDuf2uurWMrE1s=";
            dontNpmBuild = true;
            doCheck = true;
            nativeBuildInputs = [ testPython ];
            FONTCONFIG_FILE = pkgs.makeFontsConf { fontDirectories = [ pkgs.dejavu_fonts ]; };
            PLAYWRIGHT_BROWSERS_PATH = playwrightBrowsers;
            NATSUME_FIXTURE_COMMAND = "cd .. && ${testPython}/bin/python -m tests.fixture_server";
            preCheck = "cp -r ${./tests} ../tests";
            checkPhase = "runHook preCheck; npm run test:integration; runHook postCheck";
            installPhase = ''touch "$out"'';
          };

          serverSmoke =
            pkgs.runCommand "natsume-server-smoke"
              {
                nativeBuildInputs = [
                  pkgs.curl
                  testPython
                ];
              }
              ''
                cp -r ${./tests} tests
                python -c 'from pathlib import Path; from tests.database_fixture import build_search_artifact; build_search_artifact(Path("artifact"))'

                ${server}/bin/natsume-serve --artifact-dir "$PWD/artifact" --port 18000 >server.log 2>&1 &
                server_pid=$!
                trap 'cat server.log >&2; kill "$server_pid" 2>/dev/null || true' ERR
                trap 'kill "$server_pid" 2>/dev/null || true' EXIT
                for attempt in $(seq 1 50); do
                  if curl --fail --silent http://127.0.0.1:18000/api/health/ready >ready.json; then
                    break
                  fi
                  if ! kill -0 "$server_pid" 2>/dev/null; then
                    cat server.log >&2
                    exit 1
                  fi
                  sleep 0.1
                done

                curl --fail --silent http://127.0.0.1:18000/ | grep -q 'data-sveltekit-preload-data'
                curl --fail --silent \
                  'http://127.0.0.1:18000/api/collocations?term=%E6%83%85%E5%A0%B1&pos=noun&rankBy=raw' \
                  | grep -q 'fixture-build-001'
                kill "$server_pid"
                wait "$server_pid" || true
                trap - EXIT
                touch "$out"
              '';

          serverClosureCheck =
            pkgs.runCommand "natsume-server-closure-check"
              {
                exportReferencesGraph = [
                  "server-closure"
                  server
                ];
              }
              ''
                if grep -E -- '-(torch|spacy|ginza|wtpsplit|polars|nodejs|jupyter|notebook|cuda)(-|$)' \
                  server-closure; then
                  echo "server closure contains a forbidden build or accelerator dependency" >&2
                  exit 1
                fi
                touch "$out"
              '';

          builderSmoke =
            pkgs.runCommand "natsume-builder-smoke"
              {
                nativeBuildInputs = [ builderPython ];
              }
              ''
                python - <<'PY'
                from pathlib import Path
                from zipfile import ZipFile

                with ZipFile("jnlp.zip", "w") as archive:
                    archive.writestr("NLP_LATEX_CORPUS/file_DB.xls", "fixture")
                    archive.writestr("NLP_LATEX_CORPUS/V01/lesson.tex", "\\section{教材} 教材です。")
                PY
                ${corpusBuilder}/bin/natsume-corpus prepare-jnlp jnlp.zip prepared
                grep -q '教材です' prepared/NLP_LATEX_CORPUS/V01/lesson.txt
                {
                  ${pkgs.nkf}/bin/nkf --version
                  ${pkgs.pandoc}/bin/pandoc --version
                } >tool-versions.txt
                cp tool-versions.txt "$out"
              '';

          edgeConfigCheck =
            pkgs.runCommand "natsume-edge-config-check" { nativeBuildInputs = [ pkgs.nginx ]; }
              ''
                mkdir -p runtime/logs
                nginx -t -p "$PWD/runtime" -c ${./deploy/nginx.conf}
                touch "$out"
              '';

          container = pkgs.dockerTools.buildLayeredImage {
            name = "natsume-simple";
            tag = "${revision}-${system}";
            contents = [
              server
              pkgs.cacert
            ];
            extraCommands = ''
              mkdir -p etc
              echo 'natsume:x:65532:65532:Natsume service:/nonexistent:/sbin/nologin' > etc/passwd
              echo 'natsume:x:65532:' > etc/group
            '';
            config = {
              Entrypoint = [ "${server}/bin/natsume-serve" ];
              Env = [
                "NATSUME_ARTIFACT_DIR=/var/lib/natsume/artifact"
                "NATSUME_HOST=0.0.0.0"
                "NATSUME_PORT=8000"
              ];
              User = "65532:65532";
              WorkingDir = "/";
              ExposedPorts."8000/tcp" = { };
              Labels = {
                "org.opencontainers.image.revision" = revision;
                "org.opencontainers.image.source" = "https://github.com/borh/natsume-simple";
                "org.natsume.schema-version" = "1";
              };
            };
          };
        in
        {
          formatter = pkgs.nixfmt;

          pre-commit.settings.hooks = {
            nixfmt.enable = true;
            flake-checker.enable = true;
            ruff = {
              enable = true;
              entry = "${pkgs.ruff}/bin/ruff check";
            };
            ruff-format = {
              enable = true;
              entry = "${pkgs.ruff}/bin/ruff format --check";
            };
          };

          packages = {
            inherit frontend server;
            corpus-builder-cpu = corpusBuilder;
            nlp-model-integration = modelIntegration;
            default = server;
          }
          // lib.optionalAttrs (system == "x86_64-linux") {
            inherit container;
          };

          apps = {
            serve = {
              type = "app";
              program = "${server}/bin/natsume-serve";
            };
            build-corpus = {
              type = "app";
              program = "${corpusBuilder}/bin/natsume-corpus";
            };
            check = {
              type = "app";
              program = "${pkgs.writeShellScript "natsume-check" ''exec ${pkgs.nix}/bin/nix flake check "$@"''}";
            };
            default = {
              type = "app";
              program = "${server}/bin/natsume-serve";
            };
          };

          checks = {
            source-quality =
              pkgs.runCommand "natsume-source-quality"
                {
                  nativeBuildInputs = [
                    pkgs.actionlint
                    pkgs.mypy
                    pkgs.nixfmt
                    pkgs.ruff
                  ];
                }
                ''
                  cp -r ${./.} source
                  chmod -R u+w source
                  cd source
                  actionlint .github/workflows/*.yaml
                  nixfmt --check flake.nix
                  ruff format --check
                  ruff check src tests
                  mypy --ignore-missing-imports --show-error-context src
                  touch "$out"
                '';
            frontend = frontendCheck;
            backend = backendCheck;
            playwright = playwrightCheck;
            package-frontend = frontend;
            package-server = server;
            package-corpus-builder-cpu = corpusBuilder;
            builder-smoke = builderSmoke;
            edge-config = edgeConfigCheck;
            server-closure = serverClosureCheck;
            server-smoke = serverSmoke;
          }
          // lib.optionalAttrs (system == "x86_64-linux") {
            package-container = container;
          };

          devShells = {
            server = pkgs.mkShell {
              packages = [
                serverPython
                pkgs.mypy
                pkgs.ruff
                pkgs.uv
              ];
            };
            builder = pkgs.mkShell {
              packages = [
                builderPython
                pkgs.nkf
                pkgs.pandoc
                pkgs.ruff
                pkgs.uv
              ];
            };
            test = pkgs.mkShell {
              packages = [
                testPython
                pkgs.mypy
                pkgs.ruff
              ];
            };
            release = pkgs.mkShell {
              packages = [
                corpusBuilder
                server
                pkgs.curl
                pkgs.nginx
                pkgs.time
              ];
            };
            frontend = pkgs.mkShell {
              packages = [
                pkgs.nodejs
                pkgs.playwright-driver
              ];
              PLAYWRIGHT_BROWSERS_PATH = playwrightBrowsers;
            };
            default = pkgs.mkShell {
              inputsFrom = [
                config.devShells.server
                config.devShells.builder
                config.devShells.frontend
              ];
              packages = [ pkgs.git ];
            };
          };
        };
    };
}
