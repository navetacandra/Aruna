---
name: example
description: Example skill - guide for common tasks
---

# Example Skill

This skill is an example of how the agent can use guides from `.agent/skills`.

## When to use
- When the user asks for an example of skill usage
- To demonstrate skill loading via `skill_load`

## Instructions for Agent
1. If the user asks about skills, call `skill_list` to see the catalog.
2. Call `skill_load` with the skill name to get details.
3. Follow the guide in this SKILL.md step by step.
4. Always verify results with `read` / `bash` tools before answering.

## Example workflow
1. `skill_list` -> see skill `example` exists
2. `skill_load` name=example -> read this guide
3. Execute task per guide

## Notes
Add a new skill by creating a folder `.agent/skills/<name>/SKILL.md`.
SKILL.md format is free, but try to include `name` and `description` in the frontmatter.
