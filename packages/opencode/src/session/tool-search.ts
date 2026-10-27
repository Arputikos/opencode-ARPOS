import { asSchema, jsonSchema, tool, type Tool as AITool } from "ai"
import type { SessionV1 } from "@opencode-ai/core/v1/session"
import { Wildcard } from "@opencode-ai/core/util/wildcard"
import type { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { isRecord } from "@/util/record"
import { RequestFootprint } from "./request-footprint"
import DESCRIPTION from "../tool/tool-search.txt"

/**
 * ARPOS: on-demand tool loading, the way Claude Code's Tool Search works.
 *
 * Plugin tools (the whole ARPOS toolset) and MCP tools travel with every
 * request, schema and all. Once their definitions would take more than
 * `THRESHOLD` of the model's context window, they are held back: the model
 * sees their names in the system prompt and loads the ones it needs with
 * `search_tools`, which makes them real tools from the next step on.
 *
 * Which tools are loaded is read from the session's own messages — every name
 * a completed `search_tools` loaded and every tool already called — so it
 * survives a server restart, a resume and a fork with no state of its own. A
 * real compaction cuts those messages off, and the model loads them again.
 */

// Not `tool_search`: `@ai-sdk/openai` (and Azure, Bedrock, Anthropic in their
// own spellings) reserve that name for the provider's native tool search and
// replay such a call without its arguments — OpenAI then rejects the request.
export const NAME = "search_tools"
/** Share of the model's context window the deferrable definitions may take before they are deferred. */
export const THRESHOLD = 0.05
const DEFAULT_MAX_RESULTS = 5
const MAX_RESULTS = 20

export interface Entry {
  tool: AITool
  /** The MCP server the tool came from; undefined for a plugin tool. */
  server?: string
}

/** Deferred tools by name. */
export type Catalog = Record<string, Entry>

// The engine's own search tools, told apart from a plugin or MCP tool that
// happens to be keyed `search_tools` too (an MCP server keyed `search` with a
// `tools` tool) — the name alone vouches for nothing.
const searchTools = new WeakSet<AITool>()

/** The tool is this engine's `search_tools`, not a plugin or MCP tool of the same name. */
export function isSearchTool(tool: AITool) {
  return searchTools.has(tool)
}

/**
 * Split the request's tool set — already stripped of disabled tools — into
 * what is sent and what is deferred. Below the threshold everything is sent.
 * The threshold weighs every deferrable tool, loaded ones included, so loading
 * a tool never flips the mode. A plugin or MCP tool keyed `search_tools` is
 * never deferred: once anything is, the engine's search takes its name.
 */
export async function partition(input: {
  tools: Record<string, AITool>
  messages: SessionV1.WithParts[]
  model: Provider.Model
  /** Name patterns never deferred. */
  alwaysLoad: string[]
  deferrable: (tool: AITool) => boolean
  serverOf: (tool: AITool) => string | undefined
}): Promise<{ tools: Record<string, AITool>; deferred: Catalog }> {
  const unchanged = { tools: input.tools, deferred: {} }
  const candidates = Object.entries(input.tools).filter(
    ([name, tool]) =>
      name !== NAME && input.deferrable(tool) && !input.alwaysLoad.some((pattern) => Wildcard.matchName(name, pattern)),
  )
  const budget = input.model.limit.context * THRESHOLD
  if (candidates.length === 0 || budget <= 0) return unchanged

  let total = 0
  for (const [name, tool] of candidates) {
    total += await RequestFootprint.toolTokens(name, tool)
    if (total > budget) break
  }
  if (total <= budget) return unchanged

  const active = activated(input.messages)
  const deferred: Catalog = {}
  for (const [name, tool] of candidates) {
    if (active.has(name.toLowerCase())) continue
    deferred[name] = { tool, server: input.serverOf(tool) }
  }
  const tools = Object.fromEntries(Object.entries(input.tools).filter(([name]) => !(name in deferred)))
  return { tools, deferred }
}

/** Lower-cased names of the tools the session has loaded: by `search_tools`, or by calling them. */
export function activated(messages: SessionV1.WithParts[]) {
  const names = new Set<string>()
  for (const message of messages) {
    for (const part of message.parts) {
      if (part.type !== "tool") continue
      if (part.tool !== NAME) {
        if (part.state.status !== "pending") names.add(part.tool.toLowerCase())
        continue
      }
      if (part.state.status !== "completed") continue
      const loaded: unknown = part.state.metadata.loaded
      if (!Array.isArray(loaded)) continue
      for (const name of loaded) if (typeof name === "string") names.add(name.toLowerCase())
    }
  }
  return names
}

/**
 * Resolve a query against the deferred tools and the ones already loaded
 * (`loaded`: the step's other tools). `select:a,b` names tools exactly (any
 * case, no limit); anything else is keywords ranked by name, then description,
 * `+word` requiring the word in the name. Only deferred matches load anything;
 * loaded ones are reported so the model does not conclude a tool is missing.
 */
export function search(input: {
  query: string
  deferred: Catalog
  loaded: Record<string, AITool>
  maxResults?: number
}) {
  const query = input.query.trim()
  if (query.toLowerCase().startsWith("select:")) {
    const found = new Set<string>()
    const alreadyLoaded = new Set<string>()
    const missing = new Set<string>()
    for (const requested of query.slice("select:".length).split(",")) {
      const lower = requested.trim().toLowerCase()
      if (!lower) continue
      const deferred = Object.keys(input.deferred).find((name) => name.toLowerCase() === lower)
      const loaded = lower === NAME ? NAME : Object.keys(input.loaded).find((name) => name.toLowerCase() === lower)
      if (deferred) found.add(deferred)
      else if (loaded) alreadyLoaded.add(loaded)
      else missing.add(requested.trim())
    }
    return { found: [...found], alreadyLoaded: [...alreadyLoaded], missing: [...missing] }
  }

  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  const required = words.filter((word) => word.startsWith("+") && word.length > 1).map((word) => word.slice(1))
  const terms = [...required, ...words.filter((word) => !word.startsWith("+"))]
  const rank = (tools: [string, AITool][]) =>
    tools
      .map(([name, tool]) => {
        const lower = name.toLowerCase()
        if (!required.every((word) => lower.includes(word))) return { name, score: 0 }
        const segments = nameSegments(name)
        const description = (tool.description ?? "").toLowerCase()
        const score = terms.reduce((sum, term) => {
          if (lower === term) return sum + 20
          if (segments.includes(term)) return sum + 10
          if (lower.includes(term)) return sum + 5
          if (description.includes(term)) return sum + 2
          return sum
        }, 0)
        return { name, score }
      })
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
      .slice(0, limit)
      .map((item) => item.name)
  const limit = Math.min(Math.max(Math.trunc(input.maxResults ?? DEFAULT_MAX_RESULTS), 1), MAX_RESULTS)
  // Ranked apart, each capped by `max_results`: a loaded tool never pushes out one still to load.
  return {
    found: rank(Object.entries(input.deferred).map(([name, entry]) => [name, entry.tool])),
    alreadyLoaded: rank(Object.entries(input.loaded)),
    missing: [],
  }
}

function nameSegments(name: string) {
  return name
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .toLowerCase()
    .split(/[\s_\-.:]+/)
    .filter(Boolean)
}

/**
 * The `search_tools` tool for one step. `rewritten_from` is not in its schema:
 * the engine sets it when it turns a call of a deferred tool into a search for
 * that tool (`LLM` `experimental_repairToolCall`), and the result then opens
 * with the fact that the original call did not run. `shadowed` names the
 * origin of a plugin or MCP tool keyed `search_tools` that this one displaces.
 */
export function createTool(input: {
  deferred: Catalog
  loaded: Record<string, AITool>
  model: Provider.Model
  shadowed?: string
}): AITool {
  const created = tool({
    description: DESCRIPTION,
    inputSchema: jsonSchema(
      ProviderTransform.schema(input.model, {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: 'Either "select:<name>[,<name>...]" to load tools by exact name, or keywords to search for',
          },
          max_results: {
            type: "integer",
            minimum: 1,
            maximum: MAX_RESULTS,
            description: `Maximum number of keyword results (default ${DEFAULT_MAX_RESULTS})`,
          },
        },
        required: ["query"],
      }),
    ),
    async execute(args: unknown) {
      const params = isRecord(args) ? args : {}
      const query = typeof params.query === "string" ? params.query : ""
      const result = search({
        query,
        deferred: input.deferred,
        loaded: input.loaded,
        maxResults: typeof params.max_results === "number" ? params.max_results : undefined,
      })
      const lines: string[] = []
      if (typeof params.rewritten_from === "string") {
        lines.push(
          `The call to ${params.rewritten_from} did not run: the tool was not loaded yet. ` +
            `It is loaded now — call it again with arguments matching its schema below.`,
        )
      }
      if (result.found.length > 0) {
        const definitions = await Promise.all(
          result.found.map(async (name) => {
            const tool = input.deferred[name].tool
            const parameters = tool.inputSchema ? await asSchema(tool.inputSchema).jsonSchema : {}
            return `<function>${JSON.stringify({ name, description: tool.description ?? "", parameters })}</function>`
          }),
        )
        lines.push("<functions>", ...definitions, "</functions>")
        lines.push(`Loaded ${result.found.length === 1 ? "this tool" : "these tools"}; call from your next step on.`)
      }
      if (result.alreadyLoaded.length > 0)
        lines.push(`Already loaded, call directly: ${result.alreadyLoaded.join(", ")}.`)
      if (input.shadowed && result.alreadyLoaded.includes(NAME)) lines.push(shadowedNote(input.shadowed))
      if (result.missing.length > 0) lines.push(`No such tool: ${result.missing.join(", ")}.`)
      if (lines.length === 0) lines.push(`No tool matches "${query}".`)
      return {
        title: result.found.length > 0 ? `Loaded ${result.found.join(", ")}` : "No tools loaded",
        output: lines.join("\n"),
        metadata: { loaded: result.found },
      }
    },
    toModelOutput({ output }) {
      return { type: "text", value: output.output }
    },
  })
  searchTools.add(created)
  return created
}

function shadowedNote(origin: string) {
  return `The ${NAME} tool from ${origin} is unavailable while tools are deferred: its name is taken by this search tool.`
}

/** Where a tool came from, in words: `MCP server <name>` or `a plugin`. */
export function originOf(server: string | undefined) {
  return server === undefined ? "a plugin" : `MCP server ${server}`
}

/** The system prompt block naming the deferred tools, grouped by origin and sorted — stable between steps. */
export function systemBlock(deferred: Catalog, shadowed?: string) {
  const groups = new Map<string, string[]>()
  for (const [name, entry] of Object.entries(deferred)) {
    const key = entry.server ?? ""
    groups.set(key, [...(groups.get(key) ?? []), name])
  }
  const lines = [...groups]
    .sort(([a], [b]) => (a === "" ? -1 : b === "" ? 1 : a.localeCompare(b)))
    .map(
      ([server, names]) =>
        `${server === "" ? "ARPOS / plugin tools" : `MCP server ${server}`}: ${names.sort((a, b) => a.localeCompare(b)).join(", ")}`,
    )
  return [
    "<deferred-tools>",
    `Deferred tools: these tools exist, but their definitions are not loaded. To use one, load it first with ${NAME} ` +
      `(query "select:<name>" for names from this list, or keywords) — it can be called from your next step on.`,
    ...lines,
    ...(shadowed ? [shadowedNote(shadowed)] : []),
    "</deferred-tools>",
  ].join("\n")
}

export * as ToolSearch from "./tool-search"
