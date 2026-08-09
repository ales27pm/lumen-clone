import importlib.util
import plistlib
from pathlib import Path

import pytest


def load_validator():
    script = Path(__file__).resolve().parents[3] / "scripts" / "validate-msal-ios-release-config.py"
    spec = importlib.util.spec_from_file_location("validate_msal_ios_release_config", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_config(path: Path, redirect_uri: str, client_id: str = "51aa8fd9-16b2-4f8e-8b97-b8618ceb6c40") -> None:
    with path.open("wb") as handle:
        plistlib.dump(
            {
                "MSALClientID": client_id,
                "MSALRedirectURI": redirect_uri,
                "MSALAuthorityURL": "https://login.microsoftonline.com/consumers",
            },
            handle,
        )


def write_pbxproj(
    path: Path,
    bundle_id: str,
    url_scheme: str | None = None,
    msal_requirement: str = "kind = exactVersion;\n                version = 1.9.0;",
) -> None:
    debug_config = "111111111111111111111111"
    release_config = "222222222222222222222222"
    config_list = "333333333333333333333333"
    scheme = url_scheme or f"msauth.{bundle_id}"
    path.write_text(
        f"""
/* Begin XCConfigurationList section */
		{config_list} /* Build configuration list for PBXNativeTarget "Lumen" */ = {{
			isa = XCConfigurationList;
			buildConfigurations = (
				{debug_config} /* Debug */,
				{release_config} /* Release */,
			);
		}};
/* End XCConfigurationList section */

/* Begin XCBuildConfiguration section */
		{debug_config} /* Debug */ = {{
			isa = XCBuildConfiguration;
			buildSettings = {{
				INFOPLIST_KEY_CFBundleURLTypes = "{{\\n    CFBundleURLName = \\"{bundle_id}\\";\\n    CFBundleURLSchemes =     (\\n        \\"{scheme}\\"\\n    );\\n}}";
				PRODUCT_BUNDLE_IDENTIFIER = {bundle_id};
			}};
			name = Debug;
		}};
		{release_config} /* Release */ = {{
			isa = XCBuildConfiguration;
			buildSettings = {{
				INFOPLIST_KEY_CFBundleURLTypes = "{{\\n    CFBundleURLName = \\"{bundle_id}\\";\\n    CFBundleURLSchemes =     (\\n        \\"{scheme}\\"\\n    );\\n}}";
				PRODUCT_BUNDLE_IDENTIFIER = {bundle_id};
			}};
			name = Release;
		}};
/* End XCBuildConfiguration section */

/* Begin XCRemoteSwiftPackageReference section */
		A27B0C0D0E0F000000000003 /* XCRemoteSwiftPackageReference "microsoft-authentication-library-for-objc" */ = {{
			isa = XCRemoteSwiftPackageReference;
			repositoryURL = "https://github.com/AzureAD/microsoft-authentication-library-for-objc.git";
			requirement = {{
				{msal_requirement}
			}};
		}};
/* End XCRemoteSwiftPackageReference section */
""",
        encoding="utf-8",
    )


def test_validator_accepts_bundle_aligned_release_redirect(tmp_path, capsys):
    validator = load_validator()
    config = tmp_path / "MicrosoftGraphConfig.plist"
    pbxproj = tmp_path / "project.pbxproj"
    write_config(config, "msauth.com.27pm.lumenclone://auth")
    write_pbxproj(pbxproj, "com.27pm.lumenclone")
    validator.CONFIG_PLIST = config
    validator.PBXPROJ = pbxproj

    validator.main()

    output = capsys.readouterr().out
    assert "MSAL iOS release configuration validation passed" in output
    assert "com.27pm.lumenclone" in output


def test_validator_rejects_stale_redirect_bundle_mismatch(tmp_path, capsys):
    validator = load_validator()
    config = tmp_path / "MicrosoftGraphConfig.plist"
    pbxproj = tmp_path / "project.pbxproj"
    write_config(config, "msauth.com.27pm.lumen://auth")
    write_pbxproj(pbxproj, "com.27pm.lumenclone")
    validator.CONFIG_PLIST = config
    validator.PBXPROJ = pbxproj

    with pytest.raises(SystemExit) as exit_info:
        validator.main()

    assert exit_info.value.code == 1
    output = capsys.readouterr().out
    assert "MSALRedirectURI mismatch" in output
    assert "msauth.com.27pm.lumenclone://auth" in output


def test_validator_rejects_open_ended_msal_package_requirement(tmp_path, capsys):
    validator = load_validator()
    config = tmp_path / "MicrosoftGraphConfig.plist"
    pbxproj = tmp_path / "project.pbxproj"
    write_config(config, "msauth.com.27pm.lumenclone://auth")
    write_pbxproj(
        pbxproj,
        "com.27pm.lumenclone",
        msal_requirement="kind = upToNextMajorVersion;\n                minimumVersion = 1.7.0;",
    )
    validator.CONFIG_PLIST = config
    validator.PBXPROJ = pbxproj

    with pytest.raises(SystemExit) as exit_info:
        validator.main()

    assert exit_info.value.code == 1
    output = capsys.readouterr().out
    assert "must use exactVersion 1.9.0" in output
