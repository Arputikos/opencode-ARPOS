export * as Wildcard from "./wildcard"

export function match(input: string, pattern: string) {
  return toRegExp(pattern, process.platform === "win32" ? "si" : "s").test(input.replaceAll("\\", "/"))
}

// ARPOS: permission and tool names (e.g. MCP `jira_getJiraIssue`) match case-insensitively on every
// platform; values — file paths, commands — go through `match`.
export function matchName(name: string, pattern: string) {
  return toRegExp(pattern, "si").test(name.replaceAll("\\", "/"))
}

function toRegExp(pattern: string, flags: string) {
  let escaped = pattern
    .replaceAll("\\", "/")
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*/g, ".*")
    .replace(/\?/g, ".")

  if (escaped.endsWith(" .*")) escaped = escaped.slice(0, -3) + "( .*)?"

  return new RegExp("^" + escaped + "$", flags)
}
