# lib/help-system.nix
{ lib }:
let
  fmt = color: text: "\\e[${color}m${text}\\e[0m";
  bold = fmt "1";
  blue = fmt "34";
  green = fmt "32";
  yellow = fmt "33";
  cyan = fmt "36";

  # Helper to get all packages with a specific category
  getPackagesByCategory =
    packages: category:
    lib.filterAttrs (
      name: value:
      value ? passthru
      && value.passthru ? meta
      && value.passthru.meta.category == category
      && name != "default" # Exclude default package
    ) packages;

  # Helper to get uncategorized packages
  getUncategorizedPackages =
    packages:
    lib.filterAttrs (
      name: value: !(value ? passthru && value.passthru ? meta) && name != "default" # Exclude default package
    ) packages;

  # Get all unique categories, excluding Uncategorized if empty
  getCategories =
    packages:
    let
      uncategorizedPkgs = getUncategorizedPackages packages;
      hasUncategorized = (lib.length (lib.attrNames uncategorizedPkgs)) > 0;
      categories = lib.unique (
        lib.forEach (lib.attrNames packages) (
          name:
          if packages.${name} ? passthru && packages.${name}.passthru ? meta then
            packages.${name}.passthru.meta.category
          else
            "Uncategorized"
        )
      );
    in
    if hasUncategorized then categories else lib.filter (c: c != "Uncategorized") categories;

  # Format a command with description
  formatCommand =
    color: name: description:
    let
      paddingWidth = 25;
      padding = lib.fixedWidthString (paddingWidth - lib.stringLength name) " " "";
    in
    "  ${color name}${padding}- ${description}";

  # Generate help text for a single package
  formatPackageHelp =
    name: pkg:
    formatCommand green name (
      if pkg ? passthru && pkg.passthru ? meta then
        pkg.passthru.meta.description
      else
        "No description available"
    );

  # Generate help text for a category
  formatCategoryHelp =
    packages: category:
    let
      categoryPkgs = getPackagesByCategory packages category;
      pkgHelp = lib.concatStringsSep "\n" (lib.mapAttrsToList formatPackageHelp categoryPkgs);
    in
    if pkgHelp != "" then
      ''
        ${bold (category + ":")}
        ${pkgHelp}
      ''
    else
      "";

  # Common Nix commands
  getNixCommands =
    packages:
    let
      defaultPkg = packages.default or null;
      defaultCmd =
        if defaultPkg != null then
          {
            "nix run" =
              if defaultPkg ? passthru && defaultPkg.passthru ? meta then
                "Run default package (${defaultPkg.name}: ${defaultPkg.passthru.meta.description})"
              else
                "Run default package";
          }
        else
          { };
      standardCmds = {
        "nix develop" = "Enter development shell";
        "nix fmt *.nix" = "Format Nix files";
        "nix flake check" = "Check flake outputs";
        "nix build .#..." = "Build specific output";
      };
    in
    defaultCmd // standardCmds;

  # Process compose commands
  processComposeCommands = {
    "nix run .#watch" = "Start development servers (backend + frontend)";
  };

  # Format command section
  formatCommandSection =
    color: commands:
    lib.concatStringsSep "\n" (lib.mapAttrsToList (name: desc: formatCommand color name desc) commands);

in
{
  # Main function to generate help text
  generateHelpText =
    packages:
    let
      categories = getCategories packages;
      categoryHelp = lib.concatStringsSep "\n" (
        lib.filter (x: x != "") (map (formatCategoryHelp packages) categories)
      );
    in
    ''
      ${bold (blue "Available Commands:")}

      ${bold "Development Workflow:"}
      ${formatCommandSection yellow processComposeCommands}

      ${categoryHelp}

      ${bold "Nix Commands:"}
      ${formatCommandSection yellow (getNixCommands packages)}

      ${bold "Environment:"}
        ${cyan "ACCELERATOR"}      - Current accelerator: $ACCELERATOR
        ${cyan "PC_PORT_NUM"}      - Process compose port: $PC_PORT_NUM

      Type 'help' to see this overview again
    '';
}
