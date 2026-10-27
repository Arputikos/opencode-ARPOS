import { describe, expect, test } from "bun:test"
import { dynamicTool, jsonSchema, type Tool } from "ai"
import type { SessionV1 } from "@opencode-ai/core/v1/session"
import type { Provider } from "../../src/provider/provider"
import { MessageV2 } from "../../src/session/message-v2"
import { MessageID, PartID, SessionID } from "../../src/session/schema"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { ToolSearch } from "../../src/session/tool-search"
import { LLMRequestPrep } from "../../src/session/llm/request"
import type { Agent } from "../../src/agent/agent"
import type { PermissionV1 } from "@opencode-ai/core/v1/permission"

// Pure — no instance, no temp directory.

// 100k window: deferring starts above 5k tokens (~20k characters) of deferrable definitions.
const model = {
  id: "test-model",
  providerID: "test",
  api: { id: "test-model", npm: "@ai-sdk/openai-compatible" },
  limit: { context: 100_000, output: 10_000 },
} as Provider.Model

/** A tool whose serialized definition is roughly `chars` characters. */
function toolOf(chars: number, description = "x".repeat(chars)): Tool {
  return dynamicTool({
    description,
    inputSchema: jsonSchema({ type: "object", properties: { value: { type: "string" } } }),
    execute: async () => ({}),
  })
}

type Origin = "builtin" | "plugin" | { server: string }

function setup(entries: Record<string, { origin: Origin; chars?: number; description?: string }>) {
  const tools: Record<string, Tool> = {}
  const origins = new Map<Tool, Origin>()
  for (const [name, entry] of Object.entries(entries)) {
    const tool = toolOf(entry.chars ?? 100, entry.description)
    tools[name] = tool
    origins.set(tool, entry.origin)
  }
  return {
    tools,
    partition: (input: { messages?: SessionV1.WithParts[]; alwaysLoad?: string[]; tools?: Record<string, Tool> }) =>
      ToolSearch.partition({
        tools: input.tools ?? tools,
        messages: input.messages ?? [],
        model,
        alwaysLoad: input.alwaysLoad ?? [],
        deferrable: (tool) => origins.get(tool) !== "builtin",
        serverOf: (tool) => {
          const origin = origins.get(tool)
          return typeof origin === "object" ? origin.server : undefined
        },
      }),
  }
}

const sessionID = SessionID.make("ses_test")
// Which message a part sits in does not matter here — only the message order does.
const placeholder = MessageID.make("msg_placeholder")

function user(parts: SessionV1.Part[] = []): SessionV1.WithParts {
  return {
    info: {
      id: MessageID.ascending(),
      sessionID,
      role: "user",
      time: { created: 1 },
      agent: "build",
      model: { providerID: ProviderV2.ID.make("test"), modelID: ModelV2.ID.make("test-model") },
    },
    parts,
  }
}

function assistant(parts: SessionV1.Part[], extra: Partial<SessionV1.Assistant> = {}): SessionV1.WithParts {
  return {
    info: {
      id: MessageID.ascending(),
      sessionID,
      role: "assistant",
      parentID: placeholder,
      mode: "build",
      agent: "build",
      path: { cwd: "/", root: "/" },
      cost: 0,
      tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
      modelID: ModelV2.ID.make("test-model"),
      providerID: ProviderV2.ID.make("test"),
      time: { created: 1 },
      ...extra,
    },
    parts,
  }
}

function toolPart(tool: string, state: SessionV1.ToolState): SessionV1.ToolPart {
  return {
    id: PartID.ascending(),
    sessionID,
    messageID: placeholder,
    type: "tool",
    callID: PartID.ascending(),
    tool,
    state,
  }
}

function searched(loaded: string[], time: { start: number; end: number; compacted?: number } = { start: 1, end: 2 }) {
  return toolPart(ToolSearch.NAME, {
    status: "completed",
    input: {},
    output: "",
    title: "",
    metadata: { loaded },
    time,
  })
}

const heavy = {
  bash: { origin: "builtin" as const, chars: 30_000 },
  StructuredOutput: { origin: "builtin" as const },
  ask_human: { origin: "plugin" as const, chars: 5_000 },
  browser_click: { origin: "plugin" as const, chars: 5_000 },
  jira_getIssue: { origin: { server: "jira" }, chars: 10_000 },
  jira_addComment: { origin: { server: "jira" }, chars: 10_000 },
  zeta_ping: { origin: { server: "zeta" }, chars: 100 },
}

describe("ToolSearch.partition", () => {
  test("below the threshold the tool set is untouched", async () => {
    // Built-in definitions do not count: 30k characters of bash, 400 of the rest.
    const { tools, partition } = setup({
      ...heavy,
      jira_getIssue: { origin: { server: "jira" } },
      jira_addComment: { origin: { server: "jira" } },
      ask_human: { origin: "plugin" },
      browser_click: { origin: "plugin" },
    })
    const result = await partition({})
    expect(result.deferred).toEqual({})
    expect(result.tools).toBe(tools)
  })

  test("above the threshold plugin and MCP tools are deferred, built-ins and always-load patterns stay", async () => {
    const { partition } = setup(heavy)
    const result = await partition({ alwaysLoad: ["ASK_*"] })
    expect(Object.keys(result.tools).sort()).toEqual(["StructuredOutput", "ask_human", "bash"])
    expect(Object.keys(result.deferred).sort()).toEqual([
      "browser_click",
      "jira_addComment",
      "jira_getIssue",
      "zeta_ping",
    ])
    expect(result.deferred.jira_getIssue.server).toBe("jira")
    expect(result.deferred.browser_click.server).toBeUndefined()
  })

  test("a plugin or MCP tool keyed search_tools is never deferred", async () => {
    const { partition } = setup({ ...heavy, search_tools: { origin: { server: "search" }, chars: 100 } })
    const result = await partition({})
    expect(result.tools.search_tools).toBeDefined()
    expect(result.deferred.search_tools).toBeUndefined()
  })

  test("always-loaded tools do not count towards the threshold", async () => {
    const { partition } = setup({
      bash: { origin: "builtin" },
      big: { origin: { server: "s" }, chars: 30_000 },
      small: { origin: { server: "s" }, chars: 100 },
    })
    expect((await partition({ alwaysLoad: ["big"] })).deferred).toEqual({})
  })

  test("a model without a known context window defers nothing", async () => {
    const { tools } = setup(heavy)
    const result = await ToolSearch.partition({
      tools,
      messages: [],
      model: { ...model, limit: { context: 0, output: 0 } },
      alwaysLoad: [],
      deferrable: () => true,
      serverOf: () => undefined,
    })
    expect(result.deferred).toEqual({})
  })

  test("loaded tools are sent and still count, so loading never flips the mode", async () => {
    const { partition } = setup(heavy)
    // Loading both heavy jira tools leaves ~10k characters deferred — below
    // the threshold on its own — yet the mode stays on.
    const messages = [user(), assistant([searched(["JIRA_GETISSUE", "jira_addComment"])])]
    const result = await partition({ messages })
    expect(result.tools.jira_getIssue).toBeDefined()
    expect(result.tools.jira_addComment).toBeDefined()
    expect(Object.keys(result.deferred).sort()).toEqual(["ask_human", "browser_click", "zeta_ping"])
  })
})

describe("ToolSearch.activated", () => {
  test("collects names loaded by search_tools and names of tools already called", () => {
    const messages = [
      user(),
      assistant([
        searched(["jira_getIssue"]),
        toolPart("Browser_Click", { status: "error", input: {}, error: "x", time: { start: 1, end: 2 } }),
        toolPart("zeta_ping", { status: "running", input: {}, time: { start: 1 } }),
        toolPart("never_started", { status: "pending", input: {}, raw: "" }),
        toolPart(ToolSearch.NAME, {
          status: "error",
          input: {},
          error: "x",
          metadata: { loaded: ["failed"] },
          time: { start: 1, end: 2 },
        }),
      ]),
    ]
    expect([...ToolSearch.activated(messages)].sort()).toEqual(["browser_click", "jira_getissue", "zeta_ping"])
  })

  test("a pruned search_tools part keeps its activation", () => {
    const messages = [user(), assistant([searched(["jira_getIssue"], { start: 1, end: 2, compacted: 5 })])]
    expect(ToolSearch.activated(messages).has("jira_getissue")).toBe(true)
  })

  test("compaction drops activation — the model loads the tool again", async () => {
    const { partition } = setup(heavy)
    const first = user()
    const search = assistant([searched(["jira_getIssue"])], { parentID: first.info.id })
    const compaction = user([
      { id: PartID.ascending(), sessionID, messageID: placeholder, type: "compaction", auto: true },
    ])
    const summary = assistant([], { summary: true, finish: "stop", parentID: compaction.info.id })
    const next = user()

    const all = [first, search, compaction, summary, next]
    expect((await partition({ messages: all })).deferred.jira_getIssue).toBeUndefined()

    const kept = MessageV2.filterCompacted([...all].reverse())
    expect(kept.map((msg) => msg.info.id)).not.toContain(search.info.id)
    expect((await partition({ messages: kept })).deferred.jira_getIssue).toBeDefined()
  })
})

describe("ToolSearch.search", () => {
  const { tools } = setup({
    jira_getIssue: { origin: { server: "jira" }, description: "Fetch a Jira issue by key" },
    jira_addComment: { origin: { server: "jira" }, description: "Add a comment to an issue" },
    browser_click: { origin: "plugin", description: "Click an element on the page" },
    browser_type: { origin: "plugin", description: "Type text into an element" },
    notes_issue_tracker: { origin: "plugin", description: "Unrelated" },
  })
  const deferred = Object.fromEntries(Object.entries(tools).map(([name, tool]) => [name, { tool }]))

  test("select: loads exact names in any case, reports loaded and unknown ones", () => {
    expect(
      ToolSearch.search({
        query: "select:JIRA_GETISSUE, browser_type,bash,search_tools,nope",
        deferred,
        loaded: { bash: toolOf(10) },
      }),
    ).toEqual({ found: ["jira_getIssue", "browser_type"], alreadyLoaded: ["bash", "search_tools"], missing: ["nope"] })
  })

  test("select: is not capped by max_results", () => {
    const result = ToolSearch.search({
      query: "select:" + Object.keys(deferred).join(","),
      deferred,
      loaded: {},
      maxResults: 1,
    })
    expect(result.found).toHaveLength(5)
  })

  test("keywords rank a name segment above a description match", () => {
    // `issue` is a segment of jira_getIssue (camelCase) and notes_issue_tracker,
    // but only in the description of jira_addComment.
    expect(ToolSearch.search({ query: "issue", deferred, loaded: {} }).found).toEqual([
      "jira_getIssue",
      "notes_issue_tracker",
      "jira_addComment",
    ])
  })

  test("+word requires the word in the name", () => {
    expect(ToolSearch.search({ query: "+browser element", deferred, loaded: {} }).found).toEqual([
      "browser_click",
      "browser_type",
    ])
    // `click` matches nothing named jira; the required word still filters to jira tools.
    expect(ToolSearch.search({ query: "+jira click", deferred, loaded: {} }).found).toEqual([
      "jira_addComment",
      "jira_getIssue",
    ])
  })

  test("keywords report matching tools that are already loaded instead of claiming no match", () => {
    const loaded = { browser_navigate: toolOf(10, "Open a page in the browser") }
    expect(ToolSearch.search({ query: "+browser page", deferred, loaded })).toEqual({
      found: ["browser_click", "browser_type"],
      alreadyLoaded: ["browser_navigate"],
      missing: [],
    })
  })

  test("loaded matches never push deferred ones out of max_results", () => {
    // Three loaded tools outrank every deferred one on `browser`, yet both deferred matches come back.
    const loaded = {
      browser: toolOf(10, "Browser"),
      browser_navigate: toolOf(10, "Open a page"),
      browser_back: toolOf(10, "Go back"),
    }
    expect(ToolSearch.search({ query: "browser", deferred, loaded, maxResults: 2 })).toEqual({
      found: ["browser_click", "browser_type"],
      alreadyLoaded: ["browser", "browser_back"],
      missing: [],
    })
  })

  test("max_results caps keyword results", () => {
    expect(ToolSearch.search({ query: "issue", deferred, loaded: {}, maxResults: 1 }).found).toEqual(["jira_getIssue"])
  })
})

describe("ToolSearch.createTool", () => {
  const { tools } = setup({ jira_getIssue: { origin: { server: "jira" }, description: "Fetch a Jira issue" } })
  const search = ToolSearch.createTool({
    deferred: { jira_getIssue: { tool: tools.jira_getIssue, server: "jira" } },
    loaded: { bash: toolOf(10) },
    model,
  })
  const run = (input: Record<string, unknown>) =>
    search.execute!(input, { toolCallId: "call", messages: [] }) as Promise<{
      title: string
      output: string
      metadata: { loaded: string[] }
    }>

  test("returns the definition and records the loaded names", async () => {
    const result = await run({ query: "select:jira_getissue" })
    expect(result.metadata.loaded).toEqual(["jira_getIssue"])
    expect(result.output).toContain('"name":"jira_getIssue"')
    expect(result.output).toContain('"description":"Fetch a Jira issue"')
    expect(result.output).toContain('"parameters":{"type":"object","properties":{"value":{"type":"string"}}}')
    expect(result.output).not.toContain("did not run")
  })

  test("a call rewritten from a deferred tool opens with the call that did not run", async () => {
    const result = await run({ query: "select:jira_getIssue", rewritten_from: "JIRA_getIssue" })
    expect(result.output.split("\n")[0]).toBe(
      "The call to JIRA_getIssue did not run: the tool was not loaded yet. It is loaded now — call it again with arguments matching its schema below.",
    )
    expect(result.metadata.loaded).toEqual(["jira_getIssue"])
  })

  test("nothing found loads nothing", async () => {
    const result = await run({ query: "weather" })
    expect(result.metadata.loaded).toEqual([])
    expect(result.output).toBe('No tool matches "weather".')
  })

  test("is the engine's own search tool; a plugin or MCP tool named search_tools is not", () => {
    expect(ToolSearch.isSearchTool(search)).toBe(true)
    expect(ToolSearch.isSearchTool(tools.jira_getIssue)).toBe(false)
  })

  test("says which tool of the same name it displaces", async () => {
    const shadowing = ToolSearch.createTool({
      deferred: {},
      loaded: {},
      model,
      shadowed: ToolSearch.originOf("search"),
    })
    const result = (await shadowing.execute!(
      { query: "select:search_tools" },
      { toolCallId: "call", messages: [] },
    )) as {
      output: string
    }
    expect(result.output).toBe(
      "Already loaded, call directly: search_tools.\n" +
        "The search_tools tool from MCP server search is unavailable while tools are deferred: its name is taken by this search tool.",
    )
  })
})

describe("LLMRequestPrep.resolveTools", () => {
  const agent = { name: "build", mode: "primary", options: {}, permission: [] } satisfies Agent.Info
  const lastUser = user().info as SessionV1.User
  // A sub-agent narrowed to one tool: a catch-all deny, then its allow-list.
  const narrowed: PermissionV1.Ruleset = [
    { permission: "*", pattern: "*", action: "deny" },
    { permission: "jira_getIssue", pattern: "*", action: "allow" },
  ]

  test("a catch-all deny keeps the engine's search_tools — the only way to the allowed deferred tools", () => {
    const search = ToolSearch.createTool({ deferred: {}, loaded: {}, model })
    const { tools } = setup({
      jira_getIssue: { origin: { server: "jira" } },
      jira_addComment: { origin: { server: "jira" } },
    })
    const result = LLMRequestPrep.resolveTools({
      tools: { ...tools, search_tools: search },
      agent,
      permission: narrowed,
      user: lastUser,
    })
    expect(Object.keys(result).sort()).toEqual(["jira_getIssue", "search_tools"])
  })

  test("a plugin or MCP tool keyed search_tools is filtered like any other", () => {
    const foreign = toolOf(10)
    const result = LLMRequestPrep.resolveTools({
      tools: { search_tools: foreign },
      agent,
      permission: [{ permission: "search_tools", pattern: "*", action: "deny" }],
      user: lastUser,
    })
    expect(result).toEqual({})
  })
})

describe("ToolSearch.systemBlock", () => {
  test("lists plugin tools first, then each MCP server, names sorted", () => {
    const tool = toolOf(10)
    const block = ToolSearch.systemBlock({
      zeta_ping: { tool, server: "zeta" },
      browser_type: { tool },
      jira_b: { tool, server: "jira" },
      browser_click: { tool },
      jira_a: { tool, server: "jira" },
    })
    const lines = block.split("\n")
    expect(lines.slice(2, -1)).toEqual([
      "ARPOS / plugin tools: browser_click, browser_type",
      "MCP server jira: jira_a, jira_b",
      "MCP server zeta: zeta_ping",
    ])
    expect(block).toContain("search_tools")
  })
})
