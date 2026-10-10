# Project conventions

## Specifications

- Store specification files in `docs/specs/<number>-<feature-name>/`, using a three-digit, zero-padded feature number and a descriptive kebab-case name, for example `001-create-assistant-agent`.
- Number every new feature with the next number after the highest existing feature number. Keep assigned numbers stable; do not renumber existing features or reuse numbers. Include the feature number in its README title.
- Keep the feature's README and all supporting specification files together in that directory. Do not place feature specifications directly in `docs/specs/`.
- When moving specification files, update their relative links.
