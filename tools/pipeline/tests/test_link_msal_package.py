import importlib.util
from pathlib import Path


def load_linker():
    script = Path(__file__).resolve().parents[3] / "scripts" / "link-msal-package.py"
    spec = importlib.util.spec_from_file_location("link_msal_package", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def package_reference(requirement: str) -> str:
    return f'''\
\t\tA27B0C0D0E0F000000000003 /* XCRemoteSwiftPackageReference "microsoft-authentication-library-for-objc" */ = {{
\t\t\tisa = XCRemoteSwiftPackageReference;
\t\t\trepositoryURL = "https://github.com/AzureAD/microsoft-authentication-library-for-objc.git";
\t\t\trequirement = {{
{requirement}
\t\t\t}};
\t\t}};
'''


def test_enforce_exact_msal_version_replaces_open_range():
    linker = load_linker()
    source = package_reference(
        "\t\t\t\tkind = upToNextMajorVersion;\n"
        "\t\t\t\tminimumVersion = 1.7.0;"
    )

    updated = linker.enforce_exact_msal_version(source)

    assert "kind = exactVersion;" in updated
    assert "version = 1.9.0;" in updated
    assert "upToNextMajorVersion" not in updated
    assert "minimumVersion" not in updated


def test_enforce_exact_msal_version_is_idempotent():
    linker = load_linker()
    source = package_reference(
        "\t\t\t\tkind = exactVersion;\n"
        "\t\t\t\tversion = 1.9.0;"
    )

    assert linker.enforce_exact_msal_version(source) == source
