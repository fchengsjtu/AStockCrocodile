# Agent Notes

- Keep Bash scripts (`*.sh`) with LF line endings. This is especially important for `blackbox_finetune_recallXX/scripts/one_click_deploy.sh`; CRLF line endings make `bash -n` fail on Windows with errors like `$'{\r'`.
- After bulk-editing Bash scripts from Windows, normalize them back to LF and run `bash -n` before finishing.
- When updating `README.md`, group strongly related content in the same place. For example, commands, environment variables, parameters, and examples for one workflow should be documented together instead of scattered across unrelated sections.
