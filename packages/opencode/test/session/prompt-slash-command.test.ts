// Patched (arpos): a message starting with `/name` of a known command runs that
// command through `prompt` — see `promptOrCommand` in src/session/prompt.ts.
import fs from "fs/promises"
import path from "path"
import { afterAll, beforeAll, describe, expect, test } from "bun:test"
import { Effect, Layer } from "effect"
import { Bus } from "../../src/bus"
import { Command } from "../../src/command"
import { Global } from "../../src/global"
import { Instance } from "../../src/project/instance"
import { Session } from "../../src/session"
import { MessageV2 } from "../../src/session/message-v2"
import { SessionPrompt } from "../../src/session/prompt"
import { Log } from "../../src/util/log"
import { tmpdir } from "../fixture/fixture"

Log.init({ print: false })

function run<A, E>(fx: Effect.Effect<A, E, SessionPrompt.Service | Session.Service | Bus.Service>) {
  return Effect.runPromise(
    fx.pipe(
      Effect.scoped,
      Effect.provide(Layer.mergeAll(SessionPrompt.defaultLayer, Session.defaultLayer, Bus.defaultLayer)),
    ),
  )
}

function reply(text: string) {
  const chunk = (delta: object, finish?: string) =>
    `data: ${JSON.stringify({ id: "c1", object: "chat.completion.chunk", choices: [{ delta, finish_reason: finish ?? null }] })}`
  const body = [chunk({ role: "assistant" }), chunk({ content: text }), chunk({}, "stop"), "data: [DONE]"].join("\n\n") + "\n\n"
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } })
}

const requests: any[] = []
let server: ReturnType<typeof Bun.serve>

// The GLOBAL config dir, not a project: in an agent sandbox the fork cannot
// resolve a project's root (no `git` for bun's shell) and the walk upward is
// denied, so a project `opencode.json` would never load there.
const configFile = path.join(Global.Path.config, "opencode.json")
const skillDir = path.join(Global.Path.config, "skill", "tidy")
const shellSkillDir = path.join(Global.Path.config, "skill", "probe")

beforeAll(async () => {
  server = Bun.serve({
    port: 0,
    async fetch(req) {
      if (!new URL(req.url).pathname.endsWith("/chat/completions")) return new Response("not found", { status: 404 })
      requests.push(await req.json())
      return reply("done")
    },
  })
  await Bun.write(
    configFile,
    JSON.stringify({
      $schema: "https://opencode.ai/config.json",
      enabled_providers: ["alibaba"],
      provider: { alibaba: { options: { apiKey: "test-key", baseURL: `${server.url.origin}/v1` } } },
      agent: { build: { model: "alibaba/qwen-plus" } },
      command: {
        greet: { template: "Greet $ARGUMENTS politely." },
        "git/sync": { template: "Sync the branch $1." },
        chain: { template: "/greet nobody" },
        shell: { template: "Run with $ARGUMENTS" },
        shelly: { template: "Out: !`echo SHELL-RAN`" },
      },
    }),
  )
  await Bun.write(path.join(skillDir, "SKILL.md"), "---\nname: tidy\ndescription: Tidies the desk\n---\nTIDY-SKILL-BODY\n")
  await Bun.write(
    path.join(shellSkillDir, "SKILL.md"),
    "---\nname: probe\ndescription: Probes\n---\nProbe: !`echo SKILL-SHELL-RAN`\n",
  )
})

afterAll(async () => {
  server.stop(true)
  await fs.rm(configFile, { force: true })
  await fs.rm(skillDir, { recursive: true, force: true })
  await fs.rm(shellSkillDir, { recursive: true, force: true })
})

/** Sends `text` the way a client does and returns the text of the user message the session stored. */
async function send(
  text: string,
  extra: Partial<SessionPrompt.PromptInput> = {},
  permission?: { permission: string; pattern: string; action: "allow" | "deny" | "ask" }[],
) {
  await using tmp = await tmpdir()
  return Instance.provide({
    directory: tmp.path,
    fn: () =>
      run(
        Effect.gen(function* () {
          const executed: string[] = []
          const bus = yield* Bus.Service
          yield* bus.subscribeCallback(Command.Event.Executed, (event) => {
            executed.push(`${event.properties.name}|${event.properties.arguments}`)
          })
          const prompt = yield* SessionPrompt.Service
          const sessions = yield* Session.Service
          const session = yield* sessions.create(permission ? { permission } : {})
          yield* prompt.prompt({ sessionID: session.id, agent: "build", parts: [{ type: "text", text }], ...extra })
          const msgs = yield* sessions.messages({ sessionID: session.id })
          const user = msgs.find((msg) => msg.info.role === "user")!
          return {
            info: user.info as MessageV2.User,
            text: user.parts
              .filter((part): part is MessageV2.TextPart => part.type === "text")
              .map((part) => part.text)
              .join("\n"),
            executed,
          }
        }),
      ),
  })
}

describe("session.prompt — /command", () => {
  test("a known command expands its template with the arguments and is reported as executed", async () => {
    const result = await send("/greet the whole team")
    expect(result.text).toBe("Greet the whole team politely.")
    expect(result.executed).toEqual(["greet|the whole team"])
  }, 30000)

  test("Claude Code's `:` separator reaches a command in a sub-directory", async () => {
    const result = await send("/git:sync main")
    expect(result.text).toBe("Sync the branch main.")
    expect(result.executed).toEqual(["git/sync|main"])
  }, 30000)

  test("a skill runs as /name", async () => {
    const result = await send("/tidy")
    expect(result.text).toContain("TIDY-SKILL-BODY")
  }, 30000)

  test("an unknown name is an ordinary message", async () => {
    const result = await send("/workspace/desks/x/notes.md please read")
    expect(result.text).toBe("/workspace/desks/x/notes.md please read")
    expect(result.executed).toEqual([])
  }, 30000)

  test("a template starting with /other is not expanded a second time", async () => {
    const result = await send("/chain")
    expect(result.text).toBe("/greet nobody")
    expect(result.executed).toEqual(["chain|"])
  }, 30000)

  test("arguments never run as shell", async () => {
    const result = await send("/shell !`echo PWNED`")
    expect(result.text).toBe("Run with !`echo PWNED`")
  }, 30000)

  test("a command's shell block runs for an agent whose bash is unrestricted", async () => {
    const result = await send("/shelly")
    expect(result.text).toBe("Out: SHELL-RAN")
  }, 30000)

  test("a shell block never runs past a bash rule — command or skill alike", async () => {
    const disabled = [{ permission: "bash", pattern: "*", action: "deny" as const }]
    const approval = [{ permission: "bash", pattern: "rm *", action: "ask" as const }]
    const everyToolAsks = [{ permission: "*", pattern: "*", action: "ask" as const }]

    expect((await send("/shelly", {}, disabled)).text).toBe("Out: !`echo SHELL-RAN`")
    expect((await send("/shelly", {}, approval)).text).toBe("Out: !`echo SHELL-RAN`")
    expect((await send("/shelly", {}, everyToolAsks)).text).toBe("Out: !`echo SHELL-RAN`")
    expect((await send("/probe", {}, approval)).text).toContain("Probe: !`echo SKILL-SHELL-RAN`")
  }, 60000)

  test("the structured-output contract travels with the command", async () => {
    const format = { type: "json_schema" as const, schema: { type: "object" }, retryCount: 0 }
    const result = await send("/greet you", { format }).catch(() => undefined)
    // Whatever the model did with it, the turn was asked for the schema.
    expect(result === undefined || result.info.format?.type === "json_schema").toBe(true)
    expect(requests.at(-1)?.tool_choice).toBe("required")
  }, 30000)

  test("noReply keeps the text as it is", async () => {
    const result = await send("/greet you", { noReply: true })
    expect(result.text).toBe("/greet you")
  }, 30000)
})
