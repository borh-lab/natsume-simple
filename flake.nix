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
            python = pkgs.python314;
          };
          sudachipyCargoLock = pkgs.fetchurl {
            url = "https://raw.githubusercontent.com/WorksApplications/sudachi.rs/04ac6da583c30d97d6d4258f2282f6330d4e2d14/Cargo.lock";
            hash = "sha256-WMeClTKy78UGgePPz+SM5w3JgaXX2qJwYeokEdJk9DA=";
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
                      "sudachipy"
                      "toolwrapper"
                      "uctools"
                    ]
                    (
                      name:
                      prev.${name}.overrideAttrs (
                        old:
                        {
                          # These legacy sdists execute setuptools but do not
                          # declare a PEP 517 build system.
                          nativeBuildInputs =
                            (old.nativeBuildInputs or [ ])
                            ++ final.resolveBuildSystem (
                              {
                                setuptools = [ ];
                              }
                              // lib.optionalAttrs (name == "sudachipy") {
                                setuptools-rust = [ ];
                              }
                            )
                            ++ lib.optionals (name == "sudachipy") [
                              pkgs.cargo
                              pkgs.rustPlatform.cargoSetupHook
                              pkgs.rustc
                            ];
                        }
                        // lib.optionalAttrs (name == "sudachipy") {
                          # SudachiPy 0.6.10 bundles PyO3 0.23, whose version
                          # check predates CPython 3.14 support.
                          PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1";
                          postPatch = (old.postPatch or "") + ''
                            install -m 0644 ${sudachipyCargoLock} Cargo.lock
                          '';
                          cargoDeps = pkgs.rustPlatform.importCargoLock {
                            lockFile = sudachipyCargoLock;
                          };
                        }
                      )
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
          smokeFixturePython = pkgs.python314.withPackages (pythonPackages: [ pythonPackages.xlwt ]);

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
              artifact_dir="''${NATSUME_ARTIFACT_DIR:-deploy/current}"
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
                pytest -m "nlp_model and not electra_model and not rocm"
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
                mkdir -p deploy
                ln -s ../artifact deploy/current

                ${server}/bin/natsume-serve --port 18000 >server.log 2>&1 &
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
                python - <<'PY'
                import hashlib
                import json
                from pathlib import Path
                from zipfile import ZipFile

                import polars as pl
                from natsume_simple.release_inputs import canonical_article_ids_sha256

                root = Path("fixture-inputs")
                jnlp = root / "jnlp"
                (jnlp / "V01").mkdir(parents=True)
                (jnlp / "V01" / "lesson.txt").write_text("教材です。", encoding="utf-8")

                article_ids = [str(index) for index in range(971)]
                parquet = root / "train-00000-of-00015.parquet"
                pl.DataFrame(
                    {
                        "id": article_ids,
                        "url": [f"https://example.invalid/wiki/{value}" for value in article_ids],
                        "title": [f"記事 {value}" for value in article_ids],
                        "text": ["日本語の教材です。"] * len(article_ids),
                    }
                ).write_parquet(parquet)
                parquet_bytes = parquet.read_bytes()
                ted_archive = root / "ja-en.zip"
                with ZipFile(ted_archive, "w") as archive:
                    archive.writestr(
                        "ja-en/train.tags.ja-en.ja",
                        "<doc>\n<talkid>1</talkid>\n<title>教材</title>\n教材を読む。\n</doc>\n",
                    )
                ted_bytes = ted_archive.read_bytes()
                lock = {
                    "sources": [
                        {
                            "corpusId": "jnlp",
                            "status": "ready",
                            "candidate": {
                                "observedRelease": "fixture",
                                "url": "https://example.invalid/jnlp.zip",
                                "size": 1,
                                "sha256": hashlib.sha256(b"x").hexdigest(),
                            },
                        },
                        {
                            "corpusId": "wikipedia-ja-20231101",
                            "status": "ready",
                            "candidate": {
                                "urlTemplate": "https://example.invalid/{name}",
                                "verification": {
                                    "verifiedShard": parquet.name,
                                    "orderedIdentityListSha256": canonical_article_ids_sha256(article_ids),
                                },
                                "files": [
                                    {
                                        "name": parquet.name,
                                        "size": len(parquet_bytes),
                                        "sha256": hashlib.sha256(parquet_bytes).hexdigest(),
                                    }
                                ],
                            },
                        },
                        {
                            "corpusId": "ted-iwslt-2017-ja-en",
                            "servingCorpusId": "ted",
                            "status": "ready",
                            "candidate": {
                                "url": "https://example.invalid/ja-en.zip",
                                "size": len(ted_bytes),
                                "sha256": hashlib.sha256(ted_bytes).hexdigest(),
                            },
                        },
                    ]
                }
                (root / "source-lock.json").write_text(json.dumps(lock), encoding="utf-8")
                (root / "subset.json").write_text(
                    json.dumps(
                        {
                            "sourceLockCorpusId": "wikipedia-ja-20231101",
                            "articleIds": article_ids,
                        }
                    ),
                    encoding="utf-8",
                )
                PY
                ${smokeFixturePython}/bin/python - <<'PY'
                from pathlib import Path

                import xlwt

                workbook = xlwt.Workbook()
                sheet = workbook.add_sheet("Sheet1")
                headers = [
                    "ファイル名",
                    "Vol",
                    "タイトル",
                    "著者",
                    "J-Stageにおける論文URL",
                ]
                values = [
                    "lesson.tex",
                    1,
                    "教材",
                    "著者",
                    "https://example.invalid/jnlp",
                ]
                for column, (header, value) in enumerate(zip(headers, values, strict=True)):
                    sheet.write(0, column, header)
                    sheet.write(1, column, value)
                workbook.save(str(Path("fixture-inputs/jnlp/file_DB.xls")))
                PY
                ${corpusBuilder}/bin/natsume-corpus inspect-inputs \
                  --source-lock fixture-inputs/source-lock.json \
                  --wikipedia-subset fixture-inputs/subset.json \
                  --jnlp-root fixture-inputs/jnlp \
                  --wikipedia-parquet fixture-inputs/train-00000-of-00015.parquet \
                  --ted-iwslt-archive fixture-inputs/ja-en.zip \
                  >input-inspection.json
                python - <<'PY'
                import json

                with open("input-inspection.json", encoding="utf-8") as source:
                    inspection = json.load(source)
                assert inspection["wiki"]["acceptedSources"] == 971
                assert inspection["ted"] == {
                    "acceptedSources": 1,
                    "rejections": {},
                    "textUnits": 1,
                }
                PY
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
                pkgs.duckdb
                pkgs.jq
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
