let pkgs = import <nixpkgs> {};
    python38 = pkgs.python38.withPackages(ps: [ ps.pytorch-bin ps.numpy ]);
in pkgs.mkShell {
    nativeBuildInputs = [
      pkgs.cudatoolkit
    ];
    buildInputs = [
      pkgs.linuxPackages.nvidia_x11
      pkgs.cudnn
      python38
    ];
    
  shellHook = ''
    export PATH=$PATH:/run/current-system/sw/bin
  '';
}
