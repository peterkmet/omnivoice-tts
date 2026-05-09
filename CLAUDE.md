# CLAUDE.md — Instructions for Claude

## Workflow Rules
- **Never generate code without approval.** Always describe what you plan to change and wait for explicit confirmation before writing any code.
- No unsolicited code — every change requires an explicit prompt.
- Keep it simple. No over-engineering. MVP first.

## Coding Conventions
- All code, comments, and documentation in **English**.
- No external API calls. Zero cloud. Privacy by design.

## Running Tests
```bash
uv run pytest
```

## Session Start
At the beginning of every session, always read the documentation in `docs/` to refresh context on current requirements and project state.

## Compact Instructions
When compacting context, preserve in this order:
1. Architectural decisions (NEVER summarize)
2. Modified files and key changes
3. Verification state (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can be deleted, keep only pass/fail)
