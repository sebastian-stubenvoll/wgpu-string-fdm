{
  description = "Rust + PyO3 + Python venv dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        python = pkgs.python312; # IMPORTANT: more stable than 3.13 for SciPy stack

        venvDir = ".venv";

        pythonEnv = python.withPackages (ps: [
          ps.pip
          ps.setuptools
          ps.wheel
        ]);

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Rust
            rustToolchain
            pkgs.rust-analyzer
            pkgs.cargo
            pkgs.clippy

            # Python base
            pythonEnv
            pkgs.maturin

            # Native deps for scientific stack
            pkgs.stdenv.cc.cc.lib   # libstdc++.so.6 fix
            pkgs.pkg-config
            pkgs.openssl
            pkgs.zlib
            pkgs.portaudio   # needed for sounddevice
            pkgs.fftw
            pkgs.vulkan-loader # these are required so jupyter can see the gpu
            pkgs.vulkan-tools

            # system tools
            pkgs.git
          ];

          shellHook = ''
            echo "🐍 Creating Python venv in ${venvDir}..."

            if [ ! -d "${venvDir}" ]; then
              ${python}/bin/python -m venv ${venvDir}
            fi

            source ${venvDir}/bin/activate

            export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH
            export VK_LAYER_PATH=/run/opengl-driver/share/vulkan/explicit_layer.d
            export VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json

            # Upgrade pip tooling
            pip install --upgrade pip setuptools wheel

            echo "📦 Installing Python packages..."
            pip install \
              numpy \
              scipy \
              matplotlib \
              notebook \
              jupyter \
              sounddevice

            export PYO3_PYTHON=$(pwd)/${venvDir}/bin/python
            python -m ipykernel install --user \
              --name nix-wgpu \
              --display-name "Python (nix-wgpu)"

            echo "🚀 Ready: venv active + Rust + Python scientific stack loaded"
          '';
        };
      });
}
