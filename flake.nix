{
  description = "natsume-simple nix flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    flake-parts.url = "github:hercules-ci/flake-parts";

    process-compose-flake.url = "github:Platonic-Systems/process-compose-flake";
    # services-flake.url = "github:juspay/services-flake";

    git-hooks-nix = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # nix2container = {
    #   url = "github:nlewo/nix2container";
    #   inputs.nixpkgs.follows = "nixpkgs";
    # };
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        # To import a flake module
        # 1. Add foo to inputs
        # 2. Add foo as a parameter to the outputs function
        # 3. Add here: foo.flakeModule
        inputs.process-compose-flake.flakeModule
        inputs.git-hooks-nix.flakeModule
      ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        {
          config,
          self',
          inputs',
          pkgs,
          system,
          lib,
          ...
        }:
        let
          detect-accelerator = ''
            # Set default value if not already set
            : "''${ACCELERATOR:=cpu}"

            # Override with detection if not explicitly set externally
            if [ "$ACCELERATOR" = "cpu" ] && [ -z "''${ACCELERATOR_EXPLICIT:-}" ]; then
              if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
                  ACCELERATOR="cuda"
              elif [ -n "''${ROCM_PATH:-}" ] || [ -n "''${ROCM_HOME:-}" ]; then
                  ACCELERATOR="rocm"
              fi
            fi
            export ACCELERATOR
          '';
          uv-wrapped = pkgs.writeShellScriptBin "uv" ''
            ${detect-accelerator}
            exec ${pkgs.uv}/bin/uv "$@"
          '';
          uv-run = ''uv run -q --extra "''${ACCELERATOR:-cpu}"'';
          runtime-packages = [
            uv-wrapped
            pkgs.nodejs
          ];
          development-packages = [
            pkgs.bashInteractive
            pkgs.git
            pkgs.git-cliff # Changelog generator
            pkgs.bun
            pkgs.biome
            pkgs.wget
            pkgs.pandoc
            pkgs.sqlite
          ];
          help = import ./help.nix { inherit lib; };
        in
        {
          # Per-system attributes can be defined here. The self' and inputs'
          # module parameters provide easy access to attributes of the same
          # system.
          formatter = pkgs.nixfmt-rfc-style;
          pre-commit.settings.hooks = {
            nixfmt-rfc-style.enable = true;
            flake-checker.enable = true;
            ruff.enable = true;
            ruff-format.enable = true;
            biome.enable = true;
          };

          devShells = {
            default = pkgs.mkShell {
              nativeBuildInputs = development-packages ++ runtime-packages;
              shellHook =
                let
                  p = self'.packages;
                  e = pn: lib.getBin pn;
                  local-packages = map (pn: e pn) (
                    with p;
                    [
                      uv-wrapped
                      run-tests
                      lint
                      prepare-data
                      extract-patterns
                      build-frontend
                      watch-frontend
                      watch-dev-server
                      watch-prod-server
                      run-all
                    ]
                  );
                  path-string = (lib.concatStringsSep "/bin:" local-packages) + "/bin";
                  shellInit = pkgs.writeTextFile {
                    name = "shell-init";
                    text = ''
                      ${detect-accelerator}

                      # Set up shell and prompt
                      export SHELL=${pkgs.bashInteractive}/bin/bash
                      export PS1='(uv) \[\e[34m\]\w\[\e[0m\] $(if [[ $? == 0 ]]; then echo -e "\[\e[32m\]"; else echo -e "\[\e[31m\]"; fi)#\[\e[0m\] '

                      # Add local packages to PATH if not already present
                      if [[ ":$PATH:" != *":${path-string}:"* ]]; then
                        PATH="${path-string}:$PATH"
                      fi

                      eval "$(direnv hook bash)"

                      export PC_PORT_NUM=10011

                      source .venv/bin/activate
                      echo "Entering natsume-simple venv..."

                      help() {
                        cat ${pkgs.writeText "help-text" (help.generateHelpText self'.packages)}
                      }
                    '';
                  };
                in
                ''
                  ${config.pre-commit.installationScript}
                  ${config.packages.initial-setup}/bin/initial-setup
                  source ${shellInit}
                '';
            };
            # TODO: Make backend, data, and frontend-specific devShells as well
          };

          process-compose = {
            watch = {
              settings.processes = {
                backend-server.command = "${self'.packages.watch-dev-server}/bin/watch-dev-server";
                frontend-server.command = "${self'.packages.watch-frontend}/bin/watch-frontend";
              };
            };
          };

          packages.initial-setup = pkgs.writeShellApplication {
            name = "initial-setup";
            runtimeInputs = runtime-packages;
            text = ''
              ${detect-accelerator}
              export PYTHON_VERSION=3.12.7
              uv -q python install $PYTHON_VERSION
              uv -q python pin $PYTHON_VERSION
              uv -q sync --dev --extra backend --extra "$ACCELERATOR"
            '';
            passthru.meta = {
              category = "Setup";
              description = "Initialize Python environment and dependencies";
            };
          };
          packages.run-tests = pkgs.writeShellApplication {
            name = "run-tests";
            runtimeInputs = runtime-packages;
            text = ''
              ${config.packages.initial-setup}/bin/initial-setup
              mkdir -p natsume-frontend/build # Ensure directory exists to not fail test
              ${uv-run} pytest
            '';
            passthru.meta = {
              category = "Testing & QC";
              description = "Run the test suite with pytest";
            };
          };
          packages.lint = pkgs.writeShellApplication {
            name = "lint";
            runtimeInputs = runtime-packages;
            text = ''
              ${config.packages.initial-setup}/bin/initial-setup
              nix fmt {flake,help}.nix
              ${uv-run} ruff format
              ${uv-run} ruff check --fix --select I --output-format=github src notebooks tests
              ${pkgs.mypy}/bin/mypy --ignore-missing-imports src
              ${pkgs.biome}/bin/biome check --write natsume-frontend
            '';
            passthru.meta = {
              category = "Testing & QC";
              description = "Run all linters and formatters";
            };
          };
          packages.build-frontend = pkgs.writeShellApplication {
            name = "build-frontend";
            runtimeInputs = runtime-packages;
            text = ''
              ${config.packages.initial-setup}/bin/initial-setup
              cd natsume-frontend && npm i && npm run build && cd ..
            '';
            passthru.meta = {
              category = "Frontend";
              description = "Build the frontend for production";
            };
          };
          packages.watch-frontend = pkgs.writeShellApplication {
            name = "watch-frontend";
            runtimeInputs = runtime-packages;
            text = ''
              ${config.packages.initial-setup}/bin/initial-setup
              cd natsume-frontend && npm i && npm run dev && cd ..
            '';
            passthru.meta = {
              category = "Frontend";
              description = "Start frontend in development mode with hot reload";
            };
          };
          packages.watch-dev-server = pkgs.writeShellApplication {
            name = "watch-dev-server";
            runtimeInputs = runtime-packages;
            text = ''
              ${config.packages.initial-setup}/bin/initial-setup
              ${config.packages.build-frontend}/bin/build-frontend
              ${uv-run} --with fastapi --with polars fastapi dev --host localhost src/natsume_simple/server.py
            '';
            passthru.meta = {
              category = "Server";
              description = "Start backend server in development mode";
            };
          };
          packages.watch-prod-server = pkgs.writeShellApplication {
            name = "watch-prod-server";
            runtimeInputs = runtime-packages;
            text = ''
              ${config.packages.initial-setup}/bin/initial-setup
              ${config.packages.build-frontend}/bin/build-frontend
              ${uv-run} --with fastapi --with polars fastapi run --host localhost src/natsume_simple/server.py
            '';
            passthru.meta = {
              category = "Server";
              description = "Start backend server in production mode";
            };
          };
          packages.prepare-data = pkgs.writeShellApplication {
            name = "prepare-data";
            runtimeInputs = runtime-packages;
            text = ''
              ${config.packages.initial-setup}/bin/initial-setup
              ${uv-run} python src/natsume_simple/data.py --prepare
              ${uv-run} python src/natsume_simple/data.py --load \
                  --jnlp-sample-size 3000 \
                  --wiki-sample-size 3000 \
                  --ted-sample-size 30000
            '';
            passthru.meta = {
              category = "Data";
              description = "Prepare and load corpus samples";
            };
          };
          packages.extract-patterns = pkgs.writeShellApplication {
            name = "extract-patterns";
            runtimeInputs = runtime-packages;
            text = ''
              ${config.packages.initial-setup}/bin/initial-setup
              ${uv-run} python src/natsume_simple/pattern_extraction.py \
                  --input-file data/jnlp-corpus.txt \
                  --data-dir data \
                  --model ja_ginza \
                  --corpus-name "jnlp"

              ${uv-run} python src/natsume_simple/pattern_extraction.py \
                  --input-file data/ted-corpus.txt \
                  --data-dir data \
                  --model ja_ginza \
                  --corpus-name "ted"

              ${uv-run} python src/natsume_simple/pattern_extraction.py \
                  --input-file data/wiki-corpus.txt \
                  --data-dir data \
                  --model ja_ginza \
                  --corpus-name "wiki"
            '';
            passthru.meta = {
              category = "Data";
              description = "Extract patterns from all corpora";
            };
          };
          packages.run-all = pkgs.writeShellApplication {
            name = "run-all";
            runtimeInputs = runtime-packages;
            text = ''
              ${config.packages.initial-setup}/bin/initial-setup
              ${config.packages.prepare-data}/bin/prepare-data
              ${config.packages.extract-patterns}/bin/extract-patterns
              ${config.packages.watch-prod-server}/bin/watch-prod-server
            '';
            passthru.meta = {
              category = "Main";
              description = "Prepare data, extract patterns and start server";
            };
          };
          packages.default = config.packages.watch-prod-server;
        };
      flake = {
        # The usual flake attributes can be defined here, including system-
        # agnostic ones like nixosModule and system-enumerating ones, although
        # those are more easily expressed in perSystem.
      };
    };
}
