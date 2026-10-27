import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { PermissionV1 } from "@opencode-ai/core/v1/permission"
import path from "path"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import os from "os"
import { SessionID, MessageID, PartID } from "./schema"
import { MessageV2 } from "./message-v2"
import { SessionRevert } from "./revert"
import { Session } from "./session"
import { Agent } from "../agent/agent"
import { Provider } from "@/provider/provider"

import { type Tool as AITool, tool, jsonSchema } from "ai"
import type { JSONSchema7 } from "@ai-sdk/provider"
import { SessionCompaction } from "./compaction"
import { SystemPrompt } from "./system"
import { Instruction } from "./instruction"
import { Plugin } from "../plugin"
import { MAX_STEPS_PROMPT } from "@opencode-ai/core/session/runner/max-steps"
import { ToolRegistry } from "@/tool/registry"
import { MCP } from "../mcp"
import { LSP } from "@/lsp/lsp"
import { ulid } from "ulid"
import { ChildProcess, ChildProcessSpawner } from "effect/unstable/process"
import { CrossSpawnSpawner } from "@opencode-ai/core/cross-spawn-spawner"
import * as Stream from "effect/Stream"
import { Command } from "../command"
import { pathToFileURL, fileURLToPath } from "url"
import { Config } from "@/config/config"
import { ConfigMarkdown } from "@/config/markdown"
import { SessionSummary } from "./summary"
import { NamedError } from "@opencode-ai/core/util/error"
import { SessionProcessor } from "./processor"
import { RequestFootprint } from "./request-footprint"
import { ToolSearch } from "./tool-search"
import { Tool } from "@/tool/tool"
import { Permission } from "@/permission"
import { Wildcard } from "@opencode-ai/core/util/wildcard"
import { SessionStatus } from "./status"
import { LLM } from "./llm"
import { Shell } from "@opencode-ai/core/shell"
import { ShellID } from "@/tool/shell/id"
import { FSUtil } from "@opencode-ai/core/fs-util"
import { Truncate } from "@/tool/truncate"
import { Image } from "@/image/image"
import { decodeDataUrl } from "@/util/data-url"
import { Process } from "@/util/process"
import { Cause, Effect, Exit, Latch, Layer, Option, Scope, Context, Schema, Types } from "effect"
import { InstanceState } from "@/effect/instance-state"
import { TaskTool, type TaskPromptOps } from "@/tool/task"
import { SessionRunState } from "./run-state"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { EventV2Bridge } from "@/event-v2-bridge"
import { Database } from "@opencode-ai/core/database/database"
import { ModelV2 } from "@opencode-ai/core/model"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { eq } from "drizzle-orm"
import { SessionTable } from "@opencode-ai/core/session/sql"
import { SessionReminders } from "./reminders"
import { SessionTools } from "./tools"
import { LLMRequestPrep } from "./llm/request"
import { LLMEvent } from "@opencode-ai/llm"

// @ts-ignore
globalThis.AI_SDK_LOG_WARNINGS = false

const decodeMessageInfo = Schema.decodeUnknownExit(SessionV1.Info)
const decodeMessagePart = Schema.decodeUnknownExit(SessionV1.Part)
const MAX_MCP_RESOURCE_BLOB_BYTES = 10 * 1024 * 1024
const SUPPORTED_MCP_RESOURCE_ATTACHMENT_MIMES = new Set([
  "application/pdf",
  "image/gif",
  "image/jpeg",
  "image/png",
  "image/webp",
])

const STRUCTURED_OUTPUT_DESCRIPTION = `Use this tool to return your final response in the requested structured format.

IMPORTANT:
- You MUST call this tool exactly once at the end of your response
- The input must be valid JSON matching the required schema
- Complete all necessary research and tool calls BEFORE calling this tool
- This tool provides your final answer - no further actions are taken after calling it`

const STRUCTURED_OUTPUT_SYSTEM_PROMPT = `IMPORTANT: The user has requested structured output. You MUST use the StructuredOutput tool to provide your final response. Do NOT respond with plain text - you MUST call the StructuredOutput tool with your answer formatted according to the schema.`

/**
 * ARPOS: what the ARPOS orchestrator says about a submitted structured answer.
 * `text` is written for the MODEL to read, not for a log. `terminal`: no further
 * submission can change the outcome (the corrective budget is spent, the tool is
 * capped, nothing validates answers here) — the turn must end, and the
 * orchestrator decides how it ended.
 */
export interface StructuredSubmitResult {
  accepted: boolean
  terminal: boolean
  text: string
}

/**
 * ARPOS: hand the model's structured answer to the orchestrator and return its
 * verdict.
 *
 * The orchestrator owns the schema, the validation, the corrective-retry budget
 * and the transcript stamp for BOTH ARPOS engines, so this fork only relays.
 * It is the same endpoint the Claude Code MCP tool posts to, which is what makes
 * one `output_schema` behave identically on either engine.
 *
 * Fail-CLOSED on purpose: no orchestrator in the environment, or an unreachable
 * one, produces a rejection the model can see rather than a silent "accepted"
 * that nothing ever validated.
 */
async function reportStructuredOutput(sessionID: string, args: unknown): Promise<StructuredSubmitResult> {
  const baseUrl = process.env["ARPOS_ORCHESTRATOR_URL"]
  const apiKey = process.env["ARPOS_ORCHESTRATOR_API_KEY"]
  if (!baseUrl || !apiKey) {
    return {
      accepted: false,
      terminal: true,
      text:
        "Structured output could not be delivered: this OpenCode server is not connected to an ARPOS " +
        "orchestrator, so the answer cannot be validated. Stop calling StructuredOutput and end your turn.",
    }
  }
  try {
    const resp = await fetch(`${baseUrl}/api/plugin-structured-output`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ engineSessionId: sessionID, output: args }),
      signal: AbortSignal.timeout(60_000),
    })
    if (!resp.ok) {
      // A 4xx will not change on a retry (unknown session, refused token); a 5xx may.
      const terminal = resp.status < 500
      return {
        accepted: false,
        terminal,
        text: terminal
          ? `Structured output could not be validated (orchestrator returned ${resp.status}). Stop calling StructuredOutput and end your turn.`
          : `Structured output could not be validated (orchestrator returned ${resp.status}). Try calling StructuredOutput again.`,
      }
    }
    const result = (await resp.json()) as { ok?: unknown; terminal?: unknown; text?: unknown }
    if (typeof result.text !== "string") {
      return {
        accepted: false,
        terminal: false,
        text: "Structured output could not be validated: malformed orchestrator response.",
      }
    }
    return { accepted: result.ok === true, terminal: result.terminal === true, text: result.text }
  } catch (e) {
    return {
      accepted: false,
      terminal: false,
      text: `Structured output could not be delivered to the orchestrator: ${e instanceof Error ? e.message : String(e)}. Try calling StructuredOutput again.`,
    }
  }
}

function mcpResourceBase64Size(value: string) {
  const trimmed = value.replace(/\s/g, "")
  const padding = trimmed.endsWith("==") ? 2 : trimmed.endsWith("=") ? 1 : 0
  return Math.max(0, Math.floor((trimmed.length * 3) / 4) - padding)
}

function formatMcpResourceBytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${Math.ceil(value / 1024)} KB`
  return `${Math.ceil(value / (1024 * 1024))} MB`
}

function isOrphanedInterruptedTool(part: SessionV1.ToolPart) {
  // cleanup() marks abandoned tool_use blocks this way after retries/aborts.
  // They are not pending work and must not trigger an assistant-prefill request.
  return part.state.status === "error" && part.state.metadata?.interrupted === true
}

/**
 * ARPOS: how many compactions in a row the loop may start before the turn gives
 * up. "In a row" means no request in between went through without asking for
 * another compaction. 1 would be wrong — one compaction followed by a request
 * that fits is normal recovery. 2 is the smallest value that tells "compaction
 * fixed it" from "compaction is irrelevant to what overflowed": if the request
 * built from the compacted conversation needs compacting again, the excess is
 * not in the conversation (tool schemas, the system prompt, a window smaller
 * than advertised) and no further compaction will change that.
 */
const COMPACTION_STREAK_LIMIT = 2

/**
 * ARPOS: may the loop start another compaction?
 *
 * Every compaction the loop asks for goes through here — the provider (or the
 * pre-flight check) refusing a request as too large AND our own token
 * accounting deciding one is due. `fixed` is what every request carries whatever
 * the conversation holds — system prompt and tool schemas
 * (`RequestFootprint.fixed`); `threshold` is where auto-compaction kicks in,
 * undefined when it is off. When the fixed part alone reaches the threshold,
 * even the first compaction is pointless.
 */
export function nextOverflowCompaction(input: { consecutive: number; fixed: number; threshold: number | undefined }): {
  action: "compact" | "no-room" | "give-up"
  consecutive: number
} {
  if (input.threshold !== undefined && input.fixed >= input.threshold) {
    return { action: "no-room", consecutive: input.consecutive }
  }
  const consecutive = input.consecutive + 1
  return { action: consecutive >= COMPACTION_STREAK_LIMIT ? "give-up" : "compact", consecutive }
}

/**
 * ARPOS: did this step end empty in the way an oversized request does — no
 * text, no reasoning, no tool call, no file, and a finish that explains nothing
 * (`length`, `other`, `unknown`, or none)? A provider can end a request that way
 * without an error (OpenAI's `incomplete: max_output_tokens` with zero usage,
 * seen when a huge tool set came along), and the turn would otherwise end
 * "successfully" with an empty reply. Any other empty finish (`content-filter`,
 * an empty `stop`) has its own explanation and is left to the orchestrator.
 */
export function isEmptyStep(input: { parts: SessionV1.Part[]; finish: string | undefined }) {
  if (input.finish !== undefined && !UNEXPLAINED_FINISHES.has(input.finish)) return false
  return !input.parts.some((part) => {
    if (part.type === "text" || part.type === "reasoning") return part.text.length > 0
    return part.type === "tool" || part.type === "file"
  })
}

const UNEXPLAINED_FINISHES = new Set(["length", "other", "unknown"])

/**
 * ARPOS: should a queued compaction be dropped instead of run?
 *
 * A compaction queued with `overflow: true` carries a verdict that belongs to
 * ONE model: this provider refused THIS request as too large. The verdict is
 * persisted on the carrier user message and can outlive the model that earned
 * it — a credential/model swap respawns the server, the queued part is picked up
 * again, and the compaction would run under a model that never refused
 * anything, destroying history for a problem that does not exist. Dropping is
 * safe: if the new model cannot fit the request either, the provider refuses it
 * again and a fresh compaction is queued. An ordinary compaction (`overflow:
 * false` — our own token accounting) is NEVER dropped.
 */
export function shouldDropStaleOverflowCompaction(input: {
  overflow: boolean
  modelChangedSinceQueued: boolean
}): boolean {
  return input.overflow && input.modelChangedSinceQueued
}

/**
 * ARPOS: does this message CARRY an overflow compaction request that was queued
 * under a different model? `compaction.create()` records the request as a
 * `role: "user"` message whose only parts are `compaction`, pinned to the model
 * running when it was queued. That carrier reaches the model —
 * `toModelMessages` renders a compaction part as the user asking "What did we
 * do so far?" — so dropping the queued task alone is not enough. Only a message
 * that is nothing BUT compaction parts qualifies; content is never deleted on a
 * guess. Whether the compaction already ran is `findStaleOverflowCompactions`'
 * half of the decision.
 */
export function isOverflowCompactionCarrier(
  message: {
    info: { role: string; model?: { providerID: string; modelID: string } }
    parts: { type: string; overflow?: boolean }[]
  },
  currentModel: { providerID: string; modelID: string },
): boolean {
  if (message.info.role !== "user") return false
  const queued = message.info.model
  if (!queued) return false
  if (message.parts.length === 0) return false
  if (!message.parts.every((part) => part.type === "compaction")) return false
  if (!message.parts.some((part) => part.overflow === true)) return false
  return shouldDropStaleOverflowCompaction({
    overflow: true,
    modelChangedSinceQueued: queued.providerID !== currentModel.providerID || queued.modelID !== currentModel.modelID,
  })
}

export interface OverflowScanMessage {
  info: {
    id: string
    role: string
    parentID?: string
    /** `unknown`, not `boolean`: a USER message carries a summary OBJECT under the same name. */
    summary?: unknown
    finish?: string
    error?: unknown
    model?: { providerID: string; modelID: string }
  }
  parts: { type: string; overflow?: boolean }[]
}

/**
 * ARPOS: the carriers this turn must pretend were never queued.
 *
 * A carrier whose compaction ALREADY RAN is untouchable: `filterCompacted` cuts
 * the history AT it, with the `summary` assistant message right behind — so
 * removing it would hand the provider a window that opens on an assistant
 * message (Anthropic: "first message must use the 'user' role"). "Already ran"
 * is decided with `filterCompacted`'s own predicate — an assistant `summary`
 * message that finished without an error and points at the carrier.
 */
export function findStaleOverflowCompactions(
  msgs: OverflowScanMessage[],
  currentModel: { providerID: string; modelID: string },
): { id: string; queuedModel: { providerID: string; modelID: string } }[] {
  const summarised = new Set<string>()
  for (const { info } of msgs) {
    if (info.role === "assistant" && info.summary === true && info.finish && !info.error && info.parentID)
      summarised.add(info.parentID)
  }
  const stale: { id: string; queuedModel: { providerID: string; modelID: string } }[] = []
  for (const msg of msgs) {
    const queuedModel = msg.info.model
    if (!queuedModel) continue
    if (summarised.has(msg.info.id)) continue
    if (!isOverflowCompactionCarrier(msg, currentModel)) continue
    stale.push({ id: msg.info.id, queuedModel })
  }
  return stale
}

export interface Interface {
  readonly cancel: (sessionID: SessionID) => Effect.Effect<void>
  readonly prompt: (input: PromptInput) => Effect.Effect<SessionV1.WithParts, Image.Error>
  readonly loop: (input: LoopInput) => Effect.Effect<SessionV1.WithParts>
  readonly shell: (input: ShellInput) => Effect.Effect<SessionV1.WithParts, Session.BusyError>
  readonly command: (input: CommandInput) => Effect.Effect<SessionV1.WithParts, Image.Error>
  readonly resolvePromptParts: (template: string) => Effect.Effect<PromptInput["parts"]>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionPrompt") {}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const status = yield* SessionStatus.Service
    const sessions = yield* Session.Service
    const agents = yield* Agent.Service
    const provider = yield* Provider.Service
    const processor = yield* SessionProcessor.Service
    const compaction = yield* SessionCompaction.Service
    const plugin = yield* Plugin.Service
    const commands = yield* Command.Service
    const config = yield* Config.Service
    const permission = yield* Permission.Service
    const fsys = yield* FSUtil.Service
    const mcp = yield* MCP.Service
    const lsp = yield* LSP.Service
    const registry = yield* ToolRegistry.Service
    const truncate = yield* Truncate.Service
    const image = yield* Image.Service
    const spawner = yield* ChildProcessSpawner.ChildProcessSpawner
    const scope = yield* Scope.Scope
    const instruction = yield* Instruction.Service
    const state = yield* SessionRunState.Service
    const revert = yield* SessionRevert.Service
    const summary = yield* SessionSummary.Service
    const sys = yield* SystemPrompt.Service
    const llm = yield* LLM.Service
    const events = yield* EventV2Bridge.Service
    const flags = yield* RuntimeFlags.Service
    const database = yield* Database.Service
    const { db } = database
    const ops = Effect.fn("SessionPrompt.ops")(function* () {
      return {
        cancel: (sessionID: SessionID) => cancel(sessionID),
        resolvePromptParts: (template: string) => resolvePromptParts(template),
        prompt: (input: PromptInput) => prompt(input).pipe(Effect.catch(Effect.die)),
      } satisfies TaskPromptOps
    })

    const cancel = Effect.fn("SessionPrompt.cancel")(function* (sessionID: SessionID) {
      yield* Effect.logInfo("cancel", { "session.id": sessionID })
      yield* state.cancel(sessionID)
    })

    const resolvePromptParts = Effect.fn("SessionPrompt.resolvePromptParts")(function* (template: string) {
      const ctx = yield* InstanceState.context
      const parts: Types.DeepMutable<PromptInput["parts"]> = [{ type: "text", text: template }]
      const files = ConfigMarkdown.files(template)
      const seen = new Set<string>()
      yield* Effect.forEach(
        files,
        Effect.fnUntraced(function* (match) {
          const name = match[1]
          if (!name) return
          if (seen.has(name)) return
          seen.add(name)

          const filepath = name.startsWith("~/")
            ? path.join(os.homedir(), name.slice(2))
            : path.resolve(ctx.worktree, name)

          const info = yield* fsys.stat(filepath).pipe(Effect.option)
          if (Option.isNone(info)) {
            const found = yield* agents.get(name)
            if (found) parts.push({ type: "agent", name: found.name })
            return
          }
          const stat = info.value
          parts.push({
            type: "file",
            url: pathToFileURL(filepath).href,
            filename: name,
            mime: stat.type === "Directory" ? "application/x-directory" : "text/plain",
          })
        }),
        { concurrency: "unbounded", discard: true },
      )
      return parts
    })

    const title = Effect.fn("SessionPrompt.ensureTitle")(function* (input: {
      session: Session.Info
      history: SessionV1.WithParts[]
      providerID: ProviderV2.ID
      modelID: ModelV2.ID
    }) {
      if (input.session.parentID) return
      if (!Session.isDefaultTitle(input.session.title)) return

      const real = (m: SessionV1.WithParts) =>
        m.info.role === "user" && !m.parts.every((p) => "synthetic" in p && p.synthetic)
      const idx = input.history.findIndex(real)
      if (idx === -1) return
      if (input.history.filter(real).length !== 1) return

      const context = input.history.slice(0, idx + 1)
      const firstUser = context[idx]
      if (!firstUser || firstUser.info.role !== "user") return
      const firstInfo = firstUser.info

      const subtasks = firstUser.parts.filter((p): p is SessionV1.SubtaskPart => p.type === "subtask")
      const onlySubtasks = subtasks.length > 0 && firstUser.parts.every((p) => p.type === "subtask")

      const ag = yield* agents.get("title")
      if (!ag) return
      const mdl = ag.model
        ? yield* provider.getModel(ag.model.providerID, ag.model.modelID)
        : ((yield* provider.getSmallModel(input.providerID)) ??
          (yield* provider.getModel(input.providerID, input.modelID)))
      const msgs = onlySubtasks
        ? [{ role: "user" as const, content: subtasks.map((p) => p.prompt).join("\n") }]
        : yield* MessageV2.toModelMessagesEffect(context, mdl)
      const text = yield* llm
        .stream({
          agent: ag,
          user: firstInfo,
          system: [],
          small: true,
          tools: {},
          model: mdl,
          sessionID: input.session.id,
          retries: 2,
          messages: [{ role: "user", content: "Generate a title for this conversation:\n" }, ...msgs],
        })
        .pipe(
          Stream.filter(LLMEvent.is.textDelta),
          Stream.map((e) => e.text),
          Stream.mkString,
          Effect.orDie,
        )
      const cleaned = text
        .replace(/<think>[\s\S]*?<\/think>\s*/g, "")
        .split("\n")
        .map((line) => line.trim())
        .find((line) => line.length > 0)
      if (!cleaned) return
      const t = cleaned.length > 100 ? cleaned.substring(0, 97) + "..." : cleaned
      yield* sessions
        .setTitle({ sessionID: input.session.id, title: t })
        .pipe(Effect.catchCause((cause) => Effect.logError("failed to generate title", { error: Cause.squash(cause) })))
    })

    const handleSubtask = Effect.fn("SessionPrompt.handleSubtask")(function* (input: {
      task: SessionV1.SubtaskPart
      model: Provider.Model
      lastUser: SessionV1.User
      sessionID: SessionID
      session: Session.Info
      msgs: SessionV1.WithParts[]
    }) {
      const { task, model, lastUser, sessionID, session, msgs } = input
      const ctx = yield* InstanceState.context
      const promptOps = yield* ops()
      const { task: taskTool } = yield* registry.named()
      const taskModel = task.model ? yield* getModel(task.model.providerID, task.model.modelID, sessionID) : model
      const assistantMessage: SessionV1.Assistant = yield* sessions.updateMessage({
        id: MessageID.ascending(),
        role: "assistant",
        parentID: lastUser.id,
        sessionID,
        mode: task.agent,
        agent: task.agent,
        variant: lastUser.model.variant,
        path: { cwd: ctx.directory, root: ctx.worktree },
        cost: 0,
        tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
        modelID: taskModel.id,
        providerID: taskModel.providerID,
        time: { created: Date.now() },
      })
      let part: SessionV1.ToolPart = yield* sessions.updatePart({
        id: PartID.ascending(),
        messageID: assistantMessage.id,
        sessionID: assistantMessage.sessionID,
        type: "tool",
        callID: ulid(),
        tool: TaskTool.id,
        state: {
          status: "running",
          input: {
            prompt: task.prompt,
            description: task.description,
            subagent_type: task.agent,
            command: task.command,
          },
          time: { start: Date.now() },
        },
      })
      const taskArgs = {
        prompt: task.prompt,
        description: task.description,
        subagent_type: task.agent,
        command: task.command,
      }
      yield* plugin.trigger(
        "tool.execute.before",
        { tool: TaskTool.id, sessionID, callID: part.id },
        { args: taskArgs },
      )

      const taskAgent = yield* agents.get(task.agent)
      if (!taskAgent) {
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${task.agent}".${hint}` })
        yield* events.publish(Session.Event.Error, { sessionID, error: error.toObject() })
        throw error
      }

      let error: Error | undefined
      const taskAbort = new AbortController()
      const result = yield* taskTool
        .execute(taskArgs, {
          agent: task.agent,
          messageID: assistantMessage.id,
          sessionID,
          abort: taskAbort.signal,
          callID: part.callID,
          extra: { bypassAgentCheck: true, promptOps },
          messages: msgs,
          metadata: (val: { title?: string; metadata?: Record<string, any> }) =>
            Effect.gen(function* () {
              part = yield* sessions.updatePart({
                ...part,
                type: "tool",
                state: { ...part.state, ...val },
              } satisfies SessionV1.ToolPart)
            }),
          ask: (req: any) =>
            permission
              .ask({
                ...req,
                sessionID,
                ruleset: Permission.merge(taskAgent.permission, session.permission ?? []),
              })
              .pipe(Effect.orDie),
        })
        .pipe(
          Effect.catchCause((cause) => {
            const defect = Cause.squash(cause)
            error = defect instanceof Error ? defect : new Error(String(defect))
            return Effect.logError("subtask execution failed", {
              error,
              agent: task.agent,
              description: task.description,
            })
          }),
          Effect.onInterrupt(() =>
            Effect.gen(function* () {
              taskAbort.abort()
              assistantMessage.finish = "tool-calls"
              assistantMessage.time.completed = Date.now()
              yield* sessions.updateMessage(assistantMessage)
              if (part.state.status === "running") {
                yield* sessions.updatePart({
                  ...part,
                  state: {
                    status: "error",
                    error: "Cancelled",
                    time: { start: part.state.time.start, end: Date.now() },
                    metadata: part.state.metadata,
                    input: part.state.input,
                  },
                } satisfies SessionV1.ToolPart)
              }
            }),
          ),
        )

      const attachments = result?.attachments?.map((attachment) => ({
        ...attachment,
        id: PartID.ascending(),
        sessionID,
        messageID: assistantMessage.id,
      }))

      yield* plugin.trigger(
        "tool.execute.after",
        { tool: TaskTool.id, sessionID, callID: part.id, args: taskArgs },
        result,
      )

      assistantMessage.finish = "tool-calls"
      assistantMessage.time.completed = Date.now()
      yield* sessions.updateMessage(assistantMessage)

      if (result && part.state.status === "running") {
        yield* sessions.updatePart({
          ...part,
          state: {
            status: "completed",
            input: part.state.input,
            title: result.title,
            metadata: result.metadata,
            output: result.output,
            attachments,
            time: { ...part.state.time, end: Date.now() },
          },
        } satisfies SessionV1.ToolPart)
      }

      if (!result) {
        yield* sessions.updatePart({
          ...part,
          state: {
            status: "error",
            error: error ? `Tool execution failed: ${error.message}` : "Tool execution failed",
            time: {
              start: part.state.status === "running" ? part.state.time.start : Date.now(),
              end: Date.now(),
            },
            metadata: part.state.status === "pending" ? undefined : part.state.metadata,
            input: part.state.input,
          },
        } satisfies SessionV1.ToolPart)
      }

      if (!task.command) return

      const summaryUserMsg: SessionV1.User = {
        id: MessageID.ascending(),
        sessionID,
        role: "user",
        time: { created: Date.now() },
        agent: lastUser.agent,
        model: lastUser.model,
      }
      yield* sessions.updateMessage(summaryUserMsg)
      yield* sessions.updatePart({
        id: PartID.ascending(),
        messageID: summaryUserMsg.id,
        sessionID,
        type: "text",
        text: "Summarize the task tool output above and continue with your task.",
        synthetic: true,
      } satisfies SessionV1.TextPart)
    })

    const shellImpl = Effect.fn("SessionPrompt.shellImpl")(function* (input: ShellInput, ready?: Latch.Latch) {
      return yield* Effect.uninterruptibleMask((restore) =>
        Effect.gen(function* () {
          const markReady = ready ? ready.open.pipe(Effect.asVoid) : Effect.void
          const { msg, part, cwd } = yield* Effect.gen(function* () {
            const ctx = yield* InstanceState.context
            const session = yield* sessions.get(input.sessionID).pipe(Effect.orDie)
            if (session.revert) {
              yield* revert.cleanup(session)
            }
            const agent = yield* agents.get(input.agent)
            if (!agent) {
              const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
              const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
              const error = new NamedError.Unknown({ message: `Agent not found: "${input.agent}".${hint}` })
              yield* events.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
              throw error
            }
            const model = input.model ?? agent.model ?? (yield* currentModel(input.sessionID))
            const userMsg: SessionV1.User = {
              id: input.messageID ?? MessageID.ascending(),
              sessionID: input.sessionID,
              time: { created: Date.now() },
              role: "user",
              agent: input.agent,
              model: { providerID: model.providerID, modelID: model.modelID },
            }
            yield* sessions.updateMessage(userMsg)
            const userPart: SessionV1.Part = {
              type: "text",
              id: PartID.ascending(),
              messageID: userMsg.id,
              sessionID: input.sessionID,
              text: "The following tool was executed by the user",
              synthetic: true,
            }
            yield* sessions.updatePart(userPart)

            const msg: SessionV1.Assistant = {
              id: MessageID.ascending(),
              sessionID: input.sessionID,
              parentID: userMsg.id,
              mode: input.agent,
              agent: input.agent,
              cost: 0,
              path: { cwd: ctx.directory, root: ctx.worktree },
              time: { created: Date.now() },
              role: "assistant",
              tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
              modelID: model.modelID,
              providerID: model.providerID,
            }
            yield* sessions.updateMessage(msg)
            const started = Date.now()
            const part: SessionV1.ToolPart = {
              type: "tool",
              id: PartID.ascending(),
              messageID: msg.id,
              sessionID: input.sessionID,
              tool: ShellID.ToolID,
              callID: ulid(),
              state: {
                status: "running",
                time: { start: started },
                input: { command: input.command },
              },
            }
            yield* sessions.updatePart(part)
            return { msg, part, cwd: ctx.directory }
          }).pipe(Effect.ensuring(markReady))

          const cfg = yield* config.get()
          const sh = Shell.preferred(cfg.shell)
          const args = Shell.args(sh, input.command, cwd)
          let output = ""
          let aborted = false

          const finish = Effect.uninterruptible(
            Effect.gen(function* () {
              if (aborted) {
                output += "\n\n" + ["<metadata>", "User aborted the command", "</metadata>"].join("\n")
              }
              const completed = Date.now()
              if (!msg.time.completed) {
                msg.time.completed = completed
                yield* sessions.updateMessage(msg)
              }
              if (part.state.status === "running") {
                part.state = {
                  status: "completed",
                  time: { ...part.state.time, end: completed },
                  input: part.state.input,
                  title: "",
                  metadata: { output },
                  output,
                }
                yield* sessions.updatePart(part)
              }
            }),
          )

          const exit = yield* restore(
            Effect.gen(function* () {
              const shellEnv = yield* plugin.trigger(
                "shell.env",
                { cwd, sessionID: input.sessionID, callID: part.callID },
                { env: {} },
              )
              const cmd = ChildProcess.make(sh, args, {
                cwd,
                extendEnv: true,
                env: { ...shellEnv.env, TERM: "dumb" },
                stdin: "ignore",
                forceKillAfter: "3 seconds",
              })
              const handle = yield* spawner.spawn(cmd)
              yield* Stream.runForEach(Stream.decodeText(handle.all), (chunk) =>
                Effect.gen(function* () {
                  output += chunk
                  if (part.state.status === "running") {
                    part.state.metadata = { output }
                    yield* sessions.updatePart(part)
                  }
                }),
              )
              yield* handle.exitCode
            }).pipe(Effect.scoped, Effect.orDie),
          ).pipe(Effect.exit)

          if (Exit.isFailure(exit) && Cause.hasInterrupts(exit.cause) && !Cause.hasDies(exit.cause)) {
            aborted = true
          }
          yield* finish

          if (Exit.isFailure(exit) && !aborted && !Cause.hasInterruptsOnly(exit.cause)) {
            return yield* Effect.failCause(exit.cause)
          }

          return { info: msg, parts: [part] }
        }),
      )
    })

    const getModel = Effect.fn("SessionPrompt.getModel")(function* (
      providerID: ProviderV2.ID,
      modelID: ModelV2.ID,
      sessionID: SessionID,
    ) {
      const exit = yield* provider.getModel(providerID, modelID).pipe(Effect.exit)
      if (Exit.isSuccess(exit)) return exit.value
      const err = Cause.squash(exit.cause)
      if (Provider.ModelNotFoundError.isInstance(err)) {
        const hint = err.suggestions?.length ? ` Did you mean: ${err.suggestions.join(", ")}?` : ""
        yield* events.publish(Session.Event.Error, {
          sessionID,
          error: new NamedError.Unknown({
            message: `Model not found: ${err.providerID}/${err.modelID}.${hint}`,
          }).toObject(),
        })
      }
      return yield* Effect.die(err)
    })

    const currentModel = Effect.fnUntraced(function* (sessionID: SessionID) {
      const current = yield* db
        .select({ model: SessionTable.model })
        .from(SessionTable)
        .where(eq(SessionTable.id, sessionID))
        .get()
        .pipe(Effect.orDie)
      if (current?.model) {
        return {
          providerID: ProviderV2.ID.make(current.model.providerID),
          modelID: ModelV2.ID.make(current.model.id),
          ...(current.model.variant && current.model.variant !== "default" ? { variant: current.model.variant } : {}),
        }
      }
      const match = yield* sessions
        .findMessage(sessionID, (m) => m.info.role === "user" && !!m.info.model)
        .pipe(Effect.orDie)
      if (Option.isSome(match) && match.value.info.role === "user") return match.value.info.model
      return yield* provider.defaultModel().pipe(Effect.orDie)
    })

    // ARPOS: the model a turn SHOULD run on — the agent's own model if it has
    // one, else the global default (`opencode.json` `model`, which ARPOS
    // rewrites on every spawn). The pinned model of the user message (and the
    // session's stored model) goes stale when the operator changes the agent's
    // model mid-session; without this, continuing or resuming the turn re-runs
    // on the old model ("Model not found"). In memory only.
    const turnModel = Effect.fnUntraced(function* (agent: Agent.Info | undefined) {
      return agent?.model ?? (yield* provider.defaultModel().pipe(Effect.orDie))
    })

    const createUserMessage = Effect.fn("SessionPrompt.createUserMessage")(function* (input: PromptInput) {
      const agentName = input.agent
      const ag = agentName ? yield* agents.get(agentName) : yield* agents.defaultInfo()
      if (!ag) {
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${agentName}".${hint}` })
        yield* events.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }

      const model = input.model ?? ag.model ?? (yield* currentModel(input.sessionID))
      const same = ag.model && model.providerID === ag.model.providerID && model.modelID === ag.model.modelID
      const full =
        !input.variant && ag.variant && same
          ? yield* provider
              .getModel(model.providerID, model.modelID)
              .pipe(Effect.catchIf(Provider.ModelNotFoundError.isInstance, () => Effect.succeed(undefined)))
          : undefined
      const variant = input.variant ?? (ag.variant && full?.variants?.[ag.variant] ? ag.variant : undefined)

      const info: SessionV1.User = {
        id: input.messageID ?? MessageID.ascending(),
        role: "user",
        sessionID: input.sessionID,
        time: { created: Date.now() },
        tools: input.tools,
        agent: ag.name,
        model: {
          providerID: model.providerID,
          modelID: model.modelID,
          variant,
        },
        system: input.system,
        format: input.format,
      }

      const current = yield* sessions.get(input.sessionID).pipe(Effect.orDie)
      if (
        current.agent !== info.agent ||
        current.model?.providerID !== info.model.providerID ||
        current.model?.id !== info.model.modelID ||
        (current.model?.variant === "default" ? undefined : current.model?.variant) !== info.model.variant
      ) {
        yield* sessions.setAgentModel({
          sessionID: input.sessionID,
          agent: info.agent,
          model: {
            id: info.model.modelID,
            providerID: info.model.providerID,
            variant: info.model.variant ?? "default",
          },
          time: info.time.created,
        })
      }

      yield* Effect.addFinalizer(() => instruction.clear(info.id))

      type Draft<T> = T extends SessionV1.Part ? Omit<T, "id"> & { id?: string } : never
      const assign = (part: Draft<SessionV1.Part>): SessionV1.Part => ({
        ...part,
        id: part.id ? PartID.make(part.id) : PartID.ascending(),
      })

      const resolvePart: (part: PromptInput["parts"][number]) => Effect.Effect<Draft<SessionV1.Part>[]> = Effect.fn(
        "SessionPrompt.resolveUserPart",
      )(function* (part) {
        if (part.type === "file") {
          if (part.source?.type === "resource") {
            const { clientName, uri } = part.source
            yield* Effect.logInfo("mcp resource", { clientName, uri, mime: part.mime })
            const pieces: Draft<SessionV1.Part>[] = [
              {
                messageID: info.id,
                sessionID: input.sessionID,
                type: "text",
                synthetic: true,
                text: `Reading MCP resource: ${part.filename} (${uri})`,
              },
            ]
            const exit = yield* mcp.readResource(clientName, uri).pipe(Effect.exit)
            if (Exit.isSuccess(exit)) {
              const content = exit.value
              if (!content) throw new Error(`Resource not found: ${clientName}/${uri}`)
              const items = Array.isArray(content.contents) ? content.contents : [content.contents]
              for (const c of items) {
                if (!c || typeof c !== "object") continue
                if ("text" in c && typeof c.text === "string" && c.text) {
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: c.text,
                  })
                } else if ("blob" in c && typeof c.blob === "string" && c.blob) {
                  const mime = "mimeType" in c && typeof c.mimeType === "string" ? c.mimeType : part.mime
                  const filename = "uri" in c && typeof c.uri === "string" ? c.uri : part.filename
                  const size = mcpResourceBase64Size(c.blob)
                  if (!SUPPORTED_MCP_RESOURCE_ATTACHMENT_MIMES.has(mime)) {
                    pieces.push({
                      messageID: info.id,
                      sessionID: input.sessionID,
                      type: "text",
                      synthetic: true,
                      text: `[Binary MCP resource omitted: ${filename ?? uri} (${mime}, ${formatMcpResourceBytes(size)}) is not a supported attachment type]`,
                    })
                    continue
                  }
                  if (size > MAX_MCP_RESOURCE_BLOB_BYTES) {
                    pieces.push({
                      messageID: info.id,
                      sessionID: input.sessionID,
                      type: "text",
                      synthetic: true,
                      text: `[Binary MCP resource omitted: ${filename ?? uri} (${mime}, ${formatMcpResourceBytes(size)}) exceeds ${formatMcpResourceBytes(MAX_MCP_RESOURCE_BLOB_BYTES)}]`,
                    })
                    continue
                  }
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `[Binary MCP resource attached: ${filename ?? uri} (${mime})]`,
                  })
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "file",
                    mime,
                    filename,
                    url: `data:${mime};base64,${c.blob}`,
                  })
                }
              }
            } else {
              const error = Cause.squash(exit.cause)
              yield* Effect.logError("failed to read MCP resource", { error, clientName, uri })
              const message = error instanceof Error ? error.message : String(error)
              pieces.push({
                messageID: info.id,
                sessionID: input.sessionID,
                type: "text",
                synthetic: true,
                text: `Failed to read MCP resource ${part.filename}: ${message}`,
              })
            }
            return pieces
          }
          const url = new URL(part.url)
          switch (url.protocol) {
            case "data:":
              if (part.mime === "text/plain") {
                return [
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify({ filePath: part.filename })}`,
                  },
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: decodeDataUrl(part.url),
                  },
                  { ...part, messageID: info.id, sessionID: input.sessionID },
                ]
              }
              break
            case "file:": {
              yield* Effect.logInfo("file", { mime: part.mime })
              const filepath = fileURLToPath(part.url)
              const mime = (yield* fsys.isDir(filepath)) ? "application/x-directory" : part.mime

              const { read } = yield* registry.named()
              const execRead = (args: Parameters<typeof read.execute>[0], extra?: Tool.Context["extra"]) => {
                const controller = new AbortController()
                return read
                  .execute(args, {
                    sessionID: input.sessionID,
                    abort: controller.signal,
                    agent: input.agent!,
                    messageID: info.id,
                    extra: { bypassCwdCheck: true, ...extra },
                    messages: [],
                    metadata: () => Effect.void,
                    ask: () => Effect.void,
                  })
                  .pipe(Effect.onInterrupt(() => Effect.sync(() => controller.abort())))
              }

              if (mime === "text/plain") {
                let offset: number | undefined
                let limit: number | undefined
                const range = { start: url.searchParams.get("start"), end: url.searchParams.get("end") }
                if (range.start != null) {
                  const filePathURI = part.url.split("?")[0]
                  let start = parseInt(range.start)
                  let end = range.end ? parseInt(range.end) : undefined
                  if (start === end) {
                    const symbols = yield* lsp.documentSymbol(filePathURI).pipe(Effect.catch(() => Effect.succeed([])))
                    for (const symbol of symbols) {
                      let r: LSP.Range | undefined
                      if ("range" in symbol) r = symbol.range
                      else if ("location" in symbol) r = symbol.location.range
                      if (r?.start?.line && r?.start?.line === start) {
                        start = r.start.line
                        end = r?.end?.line ?? start
                        break
                      }
                    }
                  }
                  offset = Math.max(start, 1)
                  if (end) limit = end - (offset - 1)
                }
                const args = { filePath: filepath, offset, limit }
                const pieces: Draft<SessionV1.Part>[] = [
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify(args)}`,
                  },
                ]
                const exit = yield* provider.getModel(info.model.providerID, info.model.modelID).pipe(
                  Effect.flatMap((mdl) => execRead(args, { model: mdl })),
                  Effect.exit,
                )
                if (Exit.isSuccess(exit)) {
                  const result = exit.value
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: result.output,
                  })
                  if (result.attachments?.length) {
                    pieces.push(
                      ...result.attachments.map((a) => ({
                        ...a,
                        synthetic: true,
                        filename: a.filename ?? part.filename,
                        messageID: info.id,
                        sessionID: input.sessionID,
                      })),
                    )
                  } else {
                    pieces.push({ ...part, mime, messageID: info.id, sessionID: input.sessionID })
                  }
                } else {
                  const error = Cause.squash(exit.cause)
                  yield* Effect.logError("failed to read file", { error, filepath })
                  const message = error instanceof Error ? error.message : String(error)
                  yield* events.publish(Session.Event.Error, {
                    sessionID: input.sessionID,
                    error: new NamedError.Unknown({ message }).toObject(),
                  })
                  pieces.push({
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Read tool failed to read ${filepath} with the following error: ${message}`,
                  })
                }
                return pieces
              }

              if (mime === "application/x-directory") {
                const args = { filePath: filepath }
                const exit = yield* execRead(args).pipe(Effect.exit)
                if (Exit.isFailure(exit)) {
                  const error = Cause.squash(exit.cause)
                  yield* Effect.logError("failed to read directory", { error, filepath })
                  const message = error instanceof Error ? error.message : String(error)
                  yield* events.publish(Session.Event.Error, {
                    sessionID: input.sessionID,
                    error: new NamedError.Unknown({ message }).toObject(),
                  })
                  return [
                    {
                      messageID: info.id,
                      sessionID: input.sessionID,
                      type: "text",
                      synthetic: true,
                      text: `Read tool failed to read ${filepath} with the following error: ${message}`,
                    },
                  ]
                }
                return [
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: `Called the Read tool with the following input: ${JSON.stringify(args)}`,
                  },
                  {
                    messageID: info.id,
                    sessionID: input.sessionID,
                    type: "text",
                    synthetic: true,
                    text: exit.value.output,
                  },
                  { ...part, mime, messageID: info.id, sessionID: input.sessionID },
                ]
              }

              return [
                {
                  messageID: info.id,
                  sessionID: input.sessionID,
                  type: "text",
                  synthetic: true,
                  text: `Called the Read tool with the following input: {"filePath":"${filepath}"}`,
                },
                {
                  id: part.id,
                  messageID: info.id,
                  sessionID: input.sessionID,
                  type: "file",
                  url:
                    `data:${mime};base64,` +
                    Buffer.from(yield* fsys.readFile(filepath).pipe(Effect.catch(Effect.die))).toString("base64"),
                  mime,
                  filename: part.filename!,
                  source: part.source,
                },
              ]
            }
          }
        }

        if (part.type === "agent") {
          const perm = Permission.evaluate("task", part.name, ag.permission)
          const hint = perm.action === "deny" ? " . Invoked by user; guaranteed to exist." : ""
          return [
            { ...part, messageID: info.id, sessionID: input.sessionID },
            {
              messageID: info.id,
              sessionID: input.sessionID,
              type: "text",
              synthetic: true,
              text:
                " Use the above message and context to generate a prompt and call the task tool with subagent: " +
                part.name +
                hint,
            },
          ]
        }

        return [{ ...part, messageID: info.id, sessionID: input.sessionID }]
      })

      const resolvedParts = yield* Effect.forEach(input.parts, resolvePart, { concurrency: "unbounded" }).pipe(
        Effect.map((x) => x.flat().map(assign)),
      )

      yield* plugin.trigger(
        "chat.message",
        {
          sessionID: input.sessionID,
          agent: input.agent,
          model: input.model,
          messageID: input.messageID,
          variant: input.variant,
        },
        { message: info, parts: resolvedParts },
      )

      const parts = yield* Effect.forEach(resolvedParts, (part) =>
        part.type === "file" && part.mime.startsWith("image/")
          ? image.normalize(part).pipe(
              Effect.catchIf(
                (error) => error instanceof Image.ResizerUnavailableError,
                () => Effect.succeed(part),
              ),
            )
          : Effect.succeed(part),
      )

      const parsed = decodeMessageInfo(info, { errors: "all", propertyOrder: "original" })
      if (Exit.isFailure(parsed)) {
        yield* Effect.logError("invalid user message before save", {
          sessionID: input.sessionID,
          messageID: info.id,
          agent: info.agent,
          model: info.model,
          cause: Cause.pretty(parsed.cause),
        })
      }
      for (const [index, part] of parts.entries()) {
        const p = decodeMessagePart(part, { errors: "all", propertyOrder: "original" })
        if (Exit.isSuccess(p)) continue
        yield* Effect.logError("invalid user part before save", {
          sessionID: input.sessionID,
          messageID: info.id,
          partID: part.id,
          partType: part.type,
          index,
          cause: Cause.pretty(p.cause),
          part,
        })
      }

      yield* sessions.updateMessage(info)
      for (const part of parts) yield* sessions.updatePart(part)

      return { info, parts }
    }, Effect.scoped)

    const prompt: (input: PromptInput) => Effect.Effect<SessionV1.WithParts, Image.Error> = Effect.fn(
      "SessionPrompt.prompt",
    )(function* (input: PromptInput) {
      const session = yield* sessions.get(input.sessionID).pipe(Effect.orDie)
      yield* revert.cleanup(session)
      const message = yield* createUserMessage(input)
      yield* sessions.touch(input.sessionID)

      const permissions: PermissionV1.Rule[] = []
      for (const [t, enabled] of Object.entries(input.tools ?? {})) {
        permissions.push({ permission: t, action: enabled ? "allow" : "deny", pattern: "*" })
      }
      if (permissions.length > 0) {
        session.permission = permissions
        yield* sessions.setPermission({ sessionID: session.id, permission: permissions })
      }

      if (input.noReply === true) return message
      return yield* loop({ sessionID: input.sessionID })
    })

    const lastAssistant = Effect.fnUntraced(function* (sessionID: SessionID) {
      const match = yield* sessions.findMessage(sessionID, (m) => m.info.role !== "user").pipe(Effect.orDie)
      if (Option.isSome(match)) return match.value
      const msgs = yield* sessions.messages({ sessionID, limit: 1 }).pipe(Effect.orDie)
      if (msgs.length > 0) return msgs[0]
      throw new Error("Impossible")
    })

    /**
     * ARPOS: after a server restart the in-memory permission Deferreds are gone,
     * but tool parts that were mid-HITL stay in SQLite as "pending"/"running".
     * The loop alone would only render them as "[Tool execution was interrupted]"
     * and for built-in tools the model usually gives up or hallucinates a result.
     *
     * Before the first step, re-execute every such orphan on the last assistant
     * message with its stored input, through the same tool set a normal step
     * resolves (built-in, plugin and MCP tools alike). The run keeps the orphan's
     * messageID/callID, so a fresh permission.asked carries the original
     * identifiers and the orchestrator can answer it from the already-decided
     * HITL row. The result (or the real error) is written onto the SAME part.
     */
    const resumeOrphanTools = Effect.fn("SessionPrompt.resumeOrphanTools")(function* (sessionID: SessionID) {
      const msgs = yield* MessageV2.filterCompactedEffect(sessionID).pipe(
        Effect.provideService(Database.Service, database),
      )
      const last = msgs.findLast((m) => m.info.role === "assistant")
      if (!last || last.info.role !== "assistant") return
      const orphans = last.parts.filter(
        (p): p is SessionV1.ToolPart =>
          p.type === "tool" && (p.state.status === "pending" || p.state.status === "running"),
      )
      if (orphans.length === 0) return
      const lastUser = msgs.findLast((m) => m.info.role === "user")?.info
      if (!lastUser || lastUser.role !== "user") return
      const agent = yield* agents.get(lastUser.agent)
      if (!agent) return
      const target = yield* turnModel(agent)
      const model = yield* getModel(target.providerID, target.modelID, sessionID)
      const session = yield* sessions.get(sessionID).pipe(Effect.orDie)
      yield* Effect.logInfo("resumeOrphanTools", { "session.id": sessionID, count: orphans.length })

      const parts = new Map(orphans.map((part) => [part.callID, part]))
      const write = Effect.fnUntraced(function* (part: SessionV1.ToolPart) {
        const saved = yield* sessions.updatePart(part)
        parts.set(part.callID, saved)
        return saved
      })
      const settle = (callID: string, state: SessionV1.ToolPart["state"]) => {
        const part = parts.get(callID)
        return part ? write({ ...part, state }) : Effect.void
      }
      const startOf = (part: SessionV1.ToolPart) => ("time" in part.state ? part.state.time.start : Date.now())

      const tools = yield* SessionTools.resolve({
        agent,
        session,
        model,
        processor: {
          message: last.info,
          // The orphan's start was stamped before the restart; keep it.
          startToolCall: () => Effect.void,
          toolStartedAt: (callID) => {
            const part = parts.get(callID)
            return part ? startOf(part) : undefined
          },
          updateToolCall: (callID, update) => {
            const part = parts.get(callID)
            return part ? write(update(part)) : Effect.succeed(undefined)
          },
          completeToolCall: (callID, output) => {
            const part = parts.get(callID)
            if (!part) return Effect.void
            return settle(callID, {
              status: "completed",
              input: part.state.input,
              output: output.output,
              metadata: output.metadata,
              title: output.title,
              time: { start: startOf(part), end: Date.now() },
              attachments: output.attachments,
            })
          },
        },
        bypassAgentCheck: false,
        messages: msgs,
        promptOps: yield* ops(),
      }).pipe(
        Effect.provideService(Plugin.Service, plugin),
        Effect.provideService(Permission.Service, permission),
        Effect.provideService(ToolRegistry.Service, registry),
        Effect.provideService(MCP.Service, mcp),
        Effect.provideService(Truncate.Service, truncate),
        Effect.provideService(RuntimeFlags.Service, flags),
      )

      const drive = Effect.fnUntraced(function* () {
        for (const orphan of orphans) {
          const input = orphan.state.input
          const start = startOf(orphan)
          const execute = tools[orphan.tool]?.execute
          if (!execute) {
            yield* Effect.logWarning("resumeOrphanTools.missing_tool", { "session.id": sessionID, tool: orphan.tool })
            yield* settle(orphan.callID, {
              status: "error",
              input,
              error: `Tool "${orphan.tool}" is not available after server restart`,
              time: { start, end: Date.now() },
            })
            continue
          }
          yield* settle(orphan.callID, { status: "running", input, time: { start } })
          const exit = yield* Effect.promise(
            (abortSignal) =>
              Promise.resolve(
                execute(input, {
                  toolCallId: orphan.callID,
                  abortSignal,
                  messages: [],
                }),
              ) as Promise<Omit<Tool.ExecuteResult, "attachments"> & { attachments?: SessionV1.FilePart[] }>,
          ).pipe(Effect.exit)
          const current = parts.get(orphan.callID)!
          const metadata = "metadata" in current.state ? (current.state.metadata ?? {}) : {}
          if (Exit.isSuccess(exit)) {
            yield* settle(orphan.callID, {
              status: "completed",
              input,
              output: exit.value.output,
              metadata: exit.value.metadata,
              title: exit.value.title,
              time: { start, end: Date.now() },
              attachments: exit.value.attachments,
            })
            continue
          }
          const err = Cause.squash(exit.cause)
          const message = err instanceof Error ? err.message : String(err)
          yield* Effect.logWarning("resumeOrphanTools.execute_failed", {
            "session.id": sessionID,
            tool: orphan.tool,
            callID: orphan.callID,
            error: message,
          })
          yield* settle(orphan.callID, {
            status: "error",
            input,
            error: message,
            metadata,
            time: { start, end: Date.now() },
          })
        }
      })
      // Aborted (not killed): close what is still open, as the processor does, so
      // the next turn does not re-drive a call the operator stopped.
      yield* drive().pipe(
        Effect.onInterrupt(() =>
          Effect.forEach(
            [...parts.values()].filter((part) => part.state.status === "pending" || part.state.status === "running"),
            (part) =>
              settle(part.callID, {
                status: "error",
                input: part.state.input,
                error: "Tool execution aborted",
                metadata: { ...("metadata" in part.state ? part.state.metadata : {}), interrupted: true },
                time: { start: startOf(part), end: Date.now() },
              }),
            { discard: true },
          ),
        ),
      )
    })

    const runLoop: (sessionID: SessionID) => Effect.Effect<SessionV1.WithParts> = Effect.fn("SessionPrompt.run")(
      function* (sessionID: SessionID) {
        const ctx = yield* InstanceState.context
        // ARPOS: the orchestrator's verdict ended this turn's structured answer —
        // accepted, or terminally refused (budget spent, tool capped). Stopping
        // the loop is the only thing the fork still does with the verdict:
        // `toolChoice: "required"` makes the model call SOMETHING on every step,
        // so without a break it would keep calling the tool after the answer was
        // delivered — or, after a terminal refusal, forever.
        let structuredDone = false
        let step = 0
        // ARPOS: compactions started in a row, with no request in between that
        // fit. See COMPACTION_STREAK_LIMIT.
        let compactionStreak = 0
        // ARPOS: upstream keeps the loop going after an `unknown` finish. One
        // empty `unknown` step gets that retry; a second in the same turn ends it —
        // otherwise a provider that keeps answering with nothing is asked forever.
        let retriedEmptyUnknown = false
        const session = yield* sessions.get(sessionID).pipe(Effect.orDie)

        // ARPOS: end the turn on an error the orchestrator can show — recorded on
        // the message AND published, like `halt()` does for a provider error; a
        // message error alone never leaves the fork.
        const failTurn = Effect.fnUntraced(function* (
          message: SessionV1.Assistant,
          footprint: RequestFootprint.Info,
          reason: RequestFootprint.Reason,
        ) {
          const error = new SessionV1.ContextOverflowError({
            message: RequestFootprint.format(footprint, reason),
          }).toObject()
          message.error = error
          message.finish = "error"
          message.time.completed ??= Date.now()
          yield* sessions.updateMessage(message)
          yield* events.publish(Session.Event.Error, { sessionID, error })
        })

        // ARPOS: every compaction this loop starts goes through here, so that
        // none can start when it cannot help: not when the request's fixed
        // overhead alone is over the threshold, and not when the previous
        // compaction just failed to bring the request under it.
        const requestCompaction = Effect.fnUntraced(function* (input: {
          message: SessionV1.Assistant
          model: Provider.Model
          user: SessionV1.User
          footprint: RequestFootprint.Info
          rejected: boolean
        }) {
          const threshold = yield* compaction.threshold(input.model)
          const next = nextOverflowCompaction({
            consecutive: compactionStreak,
            fixed: RequestFootprint.fixed(input.footprint),
            threshold,
          })
          compactionStreak = next.consecutive
          if (next.action === "no-room" && threshold !== undefined) {
            yield* failTurn(input.message, input.footprint, { type: "no-room", threshold })
            return "break" as const
          }
          if (next.action === "give-up") {
            yield* failTurn(
              input.message,
              input.footprint,
              input.rejected
                ? { type: "rejected", times: compactionStreak }
                : { type: "compaction-loop", times: compactionStreak },
            )
            return "break" as const
          }
          yield* compaction.create({
            sessionID,
            agent: input.user.agent,
            model: input.user.model,
            auto: true,
            // Ask the processor whether the provider refused the request rather
            // than inferring it from a missing finish reason.
            overflow: input.rejected,
          })
          return "continue" as const
        })

        // ARPOS: re-drive tool calls orphaned by a server restart before the
        // first step, so the model reads a real tool result, not an interruption.
        yield* resumeOrphanTools(sessionID).pipe(
          Effect.catchCause((cause) =>
            Effect.logError("resumeOrphanTools failed", { "session.id": sessionID, cause: Cause.pretty(cause) }),
          ),
        )

        while (true) {
          yield* status.set(sessionID, { type: "busy" })
          yield* Effect.logInfo("loop", { "session.id": sessionID, step })

          let msgs = yield* MessageV2.filterCompactedEffect(sessionID).pipe(
            Effect.provideService(Database.Service, database),
          )

          let { user: lastUser, assistant: lastAssistant, finished: lastFinished, tasks } = MessageV2.latest(msgs)

          if (!lastUser) throw new Error("No user message found in stream. This should never happen.")

          const target = yield* turnModel(yield* agents.get(lastUser.agent))

          // ARPOS: drop a compaction queued under a model that is no longer the
          // one running — carrier message and all (see
          // `findStaleOverflowCompactions`). Runs BEFORE the rebind, while every
          // message still carries the model it was persisted with.
          const stale = findStaleOverflowCompactions(msgs, target)
          if (stale.length > 0) {
            const staleIds = new Set(stale.map((entry) => entry.id))
            const kept = msgs.filter((msg) => !staleIds.has(msg.info.id))
            // A turn with nothing left to answer cannot run; keeping the stale
            // request is the lesser evil.
            if (kept.some((msg) => msg.info.role === "user")) {
              for (const entry of stale) {
                yield* Effect.logInfo("dropping a stale overflow compaction", {
                  "session.id": sessionID,
                  messageID: entry.id,
                  queuedUnder: `${entry.queuedModel.providerID}/${entry.queuedModel.modelID}`,
                  runningUnder: `${target.providerID}/${target.modelID}`,
                })
              }
              msgs = kept
              ;({ user: lastUser, assistant: lastAssistant, finished: lastFinished, tasks } = MessageV2.latest(msgs))
              if (!lastUser) throw new Error("No user message found in stream. This should never happen.")
            } else {
              yield* Effect.logWarning("keeping a stale overflow compaction — it is the only user message left", {
                "session.id": sessionID,
              })
            }
          }

          if (target.providerID !== lastUser.model.providerID || target.modelID !== lastUser.model.modelID) {
            yield* Effect.logInfo("rebinding pinned user-message model to the current model", {
              "session.id": sessionID,
              from: `${lastUser.model.providerID}/${lastUser.model.modelID}`,
              to: `${target.providerID}/${target.modelID}`,
            })
            lastUser.model = { ...lastUser.model, providerID: target.providerID, modelID: target.modelID }
          }

          const lastAssistantMsg = msgs.findLast(
            (msg) => msg.info.role === "assistant" && msg.info.id === lastAssistant?.id,
          )
          // Some providers return "stop" even when the assistant message contains
          // tool calls. Keep the loop running so tool results can be sent back to
          // the model, but ignore cleanup-marked interrupted orphans.
          const hasToolCalls =
            lastAssistantMsg?.parts.some(
              (part) => part.type === "tool" && !part.metadata?.providerExecuted && !isOrphanedInterruptedTool(part),
            ) ?? false

          if (
            lastAssistant?.finish &&
            !["tool-calls", "unknown"].includes(lastAssistant.finish) &&
            !hasToolCalls &&
            lastAssistant.parentID === lastUser.id
          ) {
            const orphan = lastAssistantMsg?.parts.find(
              (part): part is SessionV1.ToolPart => part.type === "tool" && isOrphanedInterruptedTool(part),
            )
            if (orphan) {
              yield* Effect.logWarning("loop exit with orphaned interrupted tool", {
                "session.id": sessionID,
                messageID: lastAssistant.id,
                tool: orphan.tool,
                callID: orphan.callID,
              })
            }
            yield* Effect.logInfo("exiting loop", { "session.id": sessionID })
            break
          }

          step++
          if (step === 1)
            yield* title({
              session,
              modelID: lastUser.model.modelID,
              providerID: lastUser.model.providerID,
              history: msgs,
            }).pipe(Effect.ignore, Effect.forkIn(scope))

          const model = yield* getModel(lastUser.model.providerID, lastUser.model.modelID, sessionID)
          const task = tasks.pop()

          if (task?.type === "subtask") {
            yield* handleSubtask({ task, model, lastUser, sessionID, session, msgs })
            continue
          }

          if (task?.type === "compaction") {
            const result = yield* compaction.process({
              messages: msgs,
              parentID: lastUser.id,
              sessionID,
              auto: task.auto,
              overflow: task.overflow,
            })
            if (result === "stop") break
            continue
          }

          if (
            lastFinished &&
            lastFinished.summary !== true &&
            (yield* compaction.isOverflow({ tokens: lastFinished.tokens, model }))
          ) {
            yield* compaction.create({ sessionID, agent: lastUser.agent, model: lastUser.model, auto: true })
            continue
          }

          const agent = yield* agents.get(lastUser.agent)
          if (!agent) {
            const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
            const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
            const error = new NamedError.Unknown({ message: `Agent not found: "${lastUser.agent}".${hint}` })
            yield* events.publish(Session.Event.Error, { sessionID, error: error.toObject() })
            throw error
          }
          const maxSteps = agent.steps ?? Infinity
          const isLastStep = step >= maxSteps
          msgs = yield* SessionReminders.apply({ messages: msgs, agent, session }).pipe(
            Effect.provideService(RuntimeFlags.Service, flags),
            Effect.provideService(FSUtil.Service, fsys),
            Effect.provideService(Session.Service, sessions),
          )

          const msg: SessionV1.Assistant = {
            id: MessageID.ascending(),
            parentID: lastUser.id,
            role: "assistant",
            mode: agent.name,
            agent: agent.name,
            variant: lastUser.model.variant,
            path: { cwd: ctx.directory, root: ctx.worktree },
            cost: 0,
            tokens: { input: 0, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
            modelID: model.id,
            providerID: model.providerID,
            time: { created: Date.now() },
            sessionID,
          }
          yield* sessions.updateMessage(msg)

          const finalizeInterruptedAssistant = Effect.gen(function* () {
            if (msg.time.completed) return
            msg.error ??= MessageV2.fromError(new DOMException("Aborted", "AbortError"), {
              providerID: msg.providerID,
              aborted: true,
            })
            msg.time.completed = Date.now()
            yield* sessions.updateMessage(msg)
          })

          const handle = yield* processor
            .create({
              assistantMessage: msg,
              sessionID,
              model,
            })
            .pipe(Effect.onInterrupt(() => finalizeInterruptedAssistant))

          const outcome: "break" | "continue" = yield* Effect.gen(function* () {
            const lastUserMsg = msgs.findLast((m) => m.info.role === "user")
            const bypassAgentCheck = lastUserMsg?.parts.some((p) => p.type === "agent") ?? false
            const promptOps = yield* ops()
            const cfg = yield* config.get()

            const tools = yield* SessionTools.resolve({
              agent,
              session,
              model,
              processor: handle,
              bypassAgentCheck,
              messages: msgs,
              promptOps,
            }).pipe(
              Effect.provideService(Plugin.Service, plugin),
              Effect.provideService(Permission.Service, permission),
              Effect.provideService(ToolRegistry.Service, registry),
              Effect.provideService(MCP.Service, mcp),
              Effect.provideService(Truncate.Service, truncate),
              Effect.provideService(RuntimeFlags.Service, flags),
            )

            if (lastUser.format?.type === "json_schema") {
              tools[LLMRequestPrep.STRUCTURED_OUTPUT_TOOL] = createStructuredOutputTool({
                schema: lastUser.format.schema,
                submit: (args) => reportStructuredOutput(sessionID, args),
                onTurnOver() {
                  structuredDone = true
                },
              })
            }

            if (step === 1)
              yield* summary.summarize({ sessionID, messageID: lastUser.id }).pipe(Effect.ignore, Effect.forkIn(scope))

            yield* plugin.trigger("experimental.chat.messages.transform", {}, { messages: msgs })

            const [skills, env, instructions, mcpInstructions, modelMsgs] = yield* Effect.all([
              sys.skills(agent),
              sys.environment(model),
              instruction.system().pipe(Effect.orDie),
              sys.mcp(agent, session.permission),
              MessageV2.toModelMessagesEffect(msgs, model),
            ])
            const system = [
              ...env,
              ...instructions,
              ...(mcpInstructions ? [mcpInstructions] : []),
              ...(skills ? [skills] : []),
            ]
            const format = lastUser.format ?? { type: "text" as const }
            if (format.type === "json_schema") system.push(STRUCTURED_OUTPUT_SYSTEM_PROMPT)

            // ARPOS: above the threshold, plugin and MCP tools are held back
            // behind search_tools (patch 27). Here, not in SessionTools.resolve:
            // resumeOrphanTools must be able to run any tool.
            const { tools: sent, deferred } = yield* Effect.promise(() =>
              ToolSearch.partition({
                tools: LLMRequestPrep.resolveTools({ tools, agent, permission: session.permission, user: lastUser }),
                messages: msgs,
                model,
                alwaysLoad: cfg.experimental?.tool_search?.always_load ?? [],
                deferrable: SessionTools.deferrable,
                serverOf: SessionTools.serverOf,
              }),
            )
            const deferredNames = Object.keys(deferred)
            if (deferredNames.length > 0) {
              // A plugin or MCP tool keyed `search_tools` gives way to the engine's search.
              const { [ToolSearch.NAME]: displaced, ...loaded } = sent
              const shadowed = displaced && ToolSearch.originOf(SessionTools.serverOf(displaced))
              sent[ToolSearch.NAME] = ToolSearch.createTool({ deferred, loaded, model, shadowed })
              system.push(ToolSearch.systemBlock(deferred, shadowed))
            }

            // ARPOS: measure the request before sending it — the same system
            // prompt and tool set the request carries. One the model cannot
            // accept is not sent: excess in the conversation → compact (what a
            // provider rejection would lead to, minus the wasted request); tools
            // and system prompt alone too large → end the turn with the numbers,
            // naming which MCP server weighs how much.
            const footprint = yield* Effect.promise(() =>
              RequestFootprint.measure({
                model,
                system: [LLMRequestPrep.baseSystem({ agent, model, system, user: lastUser })],
                messages: modelMsgs,
                tools: sent,
                serverOf: SessionTools.serverOf,
              }),
            )
            const preflight = RequestFootprint.preflight(footprint, yield* compaction.threshold(model))
            if (preflight === "too-large") {
              yield* Effect.logWarning("request too large, not sent", {
                "session.id": sessionID,
                total: footprint.total,
                inputLimit: footprint.inputLimit,
              })
              yield* failTurn(handle.message, footprint, { type: "too-large" })
              return "break" as const
            }
            if (preflight === "compact") {
              // Never sent, so the processor never closes the message.
              handle.message.time.completed = Date.now()
              yield* sessions.updateMessage(handle.message)
              return yield* requestCompaction({
                message: handle.message,
                model,
                user: lastUser,
                footprint,
                rejected: true,
              })
            }

            const result = yield* handle.process({
              user: lastUser,
              agent,
              permission: session.permission,
              sessionID,
              parentSessionID: session.parentID,
              system,
              messages: [
                ...modelMsgs,
                ...(isLastStep ? [{ role: "assistant" as const, content: MAX_STEPS_PROMPT }] : []),
              ],
              tools: sent,
              deferredTools: deferredNames,
              model,
              toolChoice: format.type === "json_schema" ? "required" : undefined,
            })

            // ARPOS: the structured answer is settled — accepted, or refused for
            // good — so this turn is done. Nothing is written onto the message
            // here: the orchestrator's end-of-turn gate stamps the validated
            // object (or fails the turn with a StructuredOutputError) itself, so
            // a Claude Code session and an OpenCode session carry the same fields.
            if (structuredDone) {
              handle.message.finish = handle.message.finish ?? "stop"
              yield* sessions.updateMessage(handle.message)
              return "break" as const
            }

            const finished = handle.message.finish && !["tool-calls", "unknown"].includes(handle.message.finish)
            if (finished && !handle.message.error) {
              // Surface any content-filter finish (e.g. Anthropic stop_reason:
              // refusal) as an error. These turns may have produced no visible
              // output at all — previously the session went idle silently — or
              // partial text that was cut off by the provider's filter.
              if (handle.message.finish === "content-filter") {
                handle.message.error = new SessionV1.ContentFilterError({
                  message: "The response was blocked by the provider's content filter",
                }).toObject()
                yield* sessions.updateMessage(handle.message)
                yield* events.publish(Session.Event.Error, { sessionID, error: handle.message.error })
                return "break" as const
              }
            }

            // ARPOS: a step that produced nothing ends the turn with the
            // request's numbers — below the limit, a large tool set is still the
            // first suspect.
            const empty =
              result !== "compact" &&
              !handle.message.error &&
              isEmptyStep({
                parts: yield* MessageV2.parts(handle.message.id).pipe(
                  Effect.provideService(Database.Service, database),
                ),
                finish: handle.message.finish,
              })
            if (empty && handle.message.finish === "unknown" && !retriedEmptyUnknown) {
              retriedEmptyUnknown = true
            } else if (empty) {
              yield* failTurn(handle.message, RequestFootprint.withMeasuredInput(footprint, handle.message.tokens), {
                type: "empty",
                finish: handle.message.finish,
                structured: format.type === "json_schema",
              })
              return "break" as const
            }

            if (result === "stop") return "break" as const
            if (result === "compact") {
              return yield* requestCompaction({
                message: handle.message,
                model,
                user: lastUser,
                footprint: RequestFootprint.withMeasuredInput(footprint, handle.message.tokens),
                rejected: handle.overflowRejected,
              })
            }
            compactionStreak = 0
            return "continue" as const
          }).pipe(
            Effect.ensuring(instruction.clear(handle.message.id)),
            Effect.onInterrupt(() => finalizeInterruptedAssistant),
          )
          if (outcome === "break") break
          continue
        }

        yield* compaction.prune({ sessionID }).pipe(Effect.ignore, Effect.forkIn(scope))
        return yield* lastAssistant(sessionID)
      },
    )

    const loop: (input: LoopInput) => Effect.Effect<SessionV1.WithParts> = Effect.fn("SessionPrompt.loop")(function* (
      input: LoopInput,
    ) {
      return yield* state.ensureRunning(input.sessionID, lastAssistant(input.sessionID), runLoop(input.sessionID))
    })

    const shell: (input: ShellInput) => Effect.Effect<SessionV1.WithParts, Session.BusyError> = Effect.fn(
      "SessionPrompt.shell",
    )(function* (input: ShellInput) {
      const ready = yield* Latch.make()
      return yield* state.startShell(input.sessionID, lastAssistant(input.sessionID), shellImpl(input, ready), ready)
    })

    const command = Effect.fn("SessionPrompt.command")(function* (input: CommandInput) {
      yield* Effect.logInfo("command", {
        "session.id": input.sessionID,
        command: input.command,
        agent: input.agent,
      })
      const cmd = yield* commands.get(input.command)
      if (!cmd) {
        const available = (yield* commands.list()).map((c) => c.name)
        const hint = available.length ? ` Available commands: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Command not found: "${input.command}".${hint}` })
        yield* events.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }
      const agentName = cmd.agent ?? input.agent

      const agent = agentName ? yield* agents.get(agentName) : yield* agents.defaultInfo()
      if (!agent) {
        const available = (yield* agents.list()).filter((a) => !a.hidden).map((a) => a.name)
        const hint = available.length ? ` Available agents: ${available.join(", ")}` : ""
        const error = new NamedError.Unknown({ message: `Agent not found: "${agentName}".${hint}` })
        yield* events.publish(Session.Event.Error, { sessionID: input.sessionID, error: error.toObject() })
        throw error
      }

      const raw = input.arguments.match(argsRegex) ?? []
      const args = raw.map((arg) => arg.replace(quoteTrimRegex, ""))
      // ARPOS: `!`shell`` blocks are expanded in the command's own template
      // BEFORE the arguments go in — upstream expanded them after, so arguments
      // carrying a block ran it, and with a message starting with `/command`
      // routed here from `prompt` the arguments are whatever text the session
      // was sent. And only for an agent whose `bash` is unrestricted: these
      // blocks run outside the bash tool and its permission, for every source —
      // a command, a skill (its whole SKILL.md is the template) or an MCP
      // server's prompt. Any rule on `bash` that is not `allow` leaves them as text.
      let templateCommand = yield* Effect.promise(async () => cmd.template)
      const shellMatches = ConfigMarkdown.shell(templateCommand)
      if (shellMatches.length > 0) {
        const session = yield* sessions.get(input.sessionID).pipe(Effect.orDie)
        const shellAllowed = Permission.merge(agent.permission, session.permission ?? [])
          .filter((rule) => Wildcard.matchName("bash", rule.permission))
          .every((rule) => rule.action === "allow")
        if (!shellAllowed) {
          yield* Effect.logWarning("command shell blocks not run: bash is restricted", { command: input.command })
        } else {
          const cfg = yield* config.get()
          const sh = Shell.preferred(cfg.shell)
          const results = yield* Effect.promise(() =>
            Promise.all(
              shellMatches.map(async ([, cmd]) => (await Process.text([cmd], { shell: sh, nothrow: true })).text),
            ),
          )
          let index = 0
          templateCommand = templateCommand.replace(bashRegex, () => results[index++])
        }
      }

      const placeholders = templateCommand.match(placeholderRegex) ?? []
      let last = 0
      for (const item of placeholders) {
        const value = Number(item.slice(1))
        if (value > last) last = value
      }

      const withArgs = templateCommand.replaceAll(placeholderRegex, (_, index) => {
        const position = Number(index)
        const argIndex = position - 1
        if (argIndex >= args.length) return ""
        if (position === last) return args.slice(argIndex).join(" ")
        return args[argIndex]
      })
      const usesArgumentsPlaceholder = templateCommand.includes("$ARGUMENTS")
      let template = withArgs.replaceAll("$ARGUMENTS", input.arguments)

      if (placeholders.length === 0 && !usesArgumentsPlaceholder && input.arguments.trim()) {
        template = template + "\n\n" + input.arguments
      }
      template = template.trim()

      const taskModel = yield* Effect.gen(function* () {
        if (cmd.model) return Provider.parseModel(cmd.model)
        if (cmd.agent) {
          const cmdAgent = yield* agents.get(cmd.agent)
          if (cmdAgent?.model) return cmdAgent.model
        }
        if (input.model) return Provider.parseModel(input.model)
        return yield* currentModel(input.sessionID)
      })

      yield* getModel(taskModel.providerID, taskModel.modelID, input.sessionID)

      const templateParts = yield* resolvePromptParts(template)
      const inputFiles = new Set(
        input.parts?.filter((part) => new URL(part.url).protocol === "file:").map((part) => fileURLToPath(part.url)),
      )
      const uniqueTemplateParts = templateParts.filter(
        (part) => part.type !== "file" || !inputFiles.has(fileURLToPath(part.url)),
      )
      const isSubtask = (agent.mode === "subagent" && cmd.subtask !== false) || cmd.subtask === true
      const parts = isSubtask
        ? [
            {
              type: "subtask" as const,
              agent: agent.name,
              description: cmd.description ?? "",
              command: input.command,
              model: { providerID: taskModel.providerID, modelID: taskModel.modelID },
              prompt: templateParts.find((y) => y.type === "text")?.text ?? "",
            },
          ]
        : [...uniqueTemplateParts, ...(input.parts ?? [])]

      const userAgent = isSubtask ? (input.agent ?? (yield* agents.defaultInfo()).name) : agent.name
      const userModel = isSubtask
        ? input.model
          ? Provider.parseModel(input.model)
          : yield* currentModel(input.sessionID)
        : taskModel

      yield* plugin.trigger(
        "command.execute.before",
        { command: input.command, sessionID: input.sessionID, arguments: input.arguments },
        { parts },
      )

      const result = yield* prompt({
        sessionID: input.sessionID,
        messageID: input.messageID,
        model: userModel,
        agent: userAgent,
        parts,
        variant: input.variant,
        format: input.format,
      })
      yield* events.publish(Command.Event.Executed, {
        name: input.command,
        sessionID: input.sessionID,
        arguments: input.arguments,
        messageID: result.info.id,
      })
      return result
    })

    /**
     * ARPOS: a message whose text starts with `/name` of a known command (a
     * command, a skill, an MCP prompt) runs that command — what `claude -p`
     * does with the same text. `session.command` cannot carry the
     * structured-output contract and nobody types into its endpoint: a human or
     * a trigger sends a message. `:` reads as `/`, so `/git:sync` (Claude Code's
     * spelling) runs `git/sync`. An unknown name (`/workspace/x`) is an ordinary
     * message. Only the entry point routes: `command` and the task tool call the
     * inner `prompt`, so a template starting with `/other` never expands twice.
     */
    const promptOrCommand: (input: PromptInput) => Effect.Effect<SessionV1.WithParts, Image.Error> = Effect.fn(
      "SessionPrompt.promptOrCommand",
    )(function* (input: PromptInput) {
      const slash = yield* slashCommand(input)
      if (!slash) return yield* prompt(input)
      return yield* command({
        sessionID: input.sessionID,
        messageID: input.messageID,
        agent: input.agent,
        model: input.model ? `${input.model.providerID}/${input.model.modelID}` : undefined,
        variant: input.variant,
        format: input.format,
        command: slash.name,
        arguments: slash.arguments,
        parts: slash.files,
      })
    })

    const slashCommand = Effect.fnUntraced(function* (input: PromptInput) {
      if (input.noReply || input.system !== undefined || input.tools !== undefined) return undefined
      const [first, ...rest] = input.parts
      if (first?.type !== "text" || first.synthetic) return undefined
      const files = rest.filter((part) => part.type === "file")
      if (files.length !== rest.length) return undefined
      const match = first.text.match(slashRegex)
      if (!match) return undefined
      const name = match[1]
      for (const candidate of [name, name.replaceAll(":", "/")]) {
        if (yield* commands.get(candidate)) {
          return { name: candidate, arguments: match[2] ?? "", files: files.length > 0 ? files : undefined }
        }
      }
      return undefined
    })

    return Service.of({
      cancel,
      prompt: promptOrCommand,
      loop,
      shell,
      command,
      resolvePromptParts,
    })
  }),
)

const ModelRef = Schema.Struct({
  providerID: ProviderV2.ID,
  modelID: ModelV2.ID,
})

export const PromptInput = Schema.Struct({
  sessionID: SessionID,
  messageID: Schema.optional(MessageID),
  model: Schema.optional(ModelRef),
  agent: Schema.optional(Schema.String),
  noReply: Schema.optional(Schema.Boolean),
  tools: Schema.optional(Schema.Record(Schema.String, Schema.Boolean)).annotate({
    description:
      "@deprecated tools and permissions have been merged, you can set permissions on the session itself now",
  }),
  format: Schema.optional(SessionV1.Format),
  system: Schema.optional(Schema.String),
  variant: Schema.optional(Schema.String),
  parts: Schema.Array(
    Schema.Union([
      SessionV1.TextPartInput,
      SessionV1.FilePartInput,
      SessionV1.AgentPartInput,
      SessionV1.SubtaskPartInput,
    ]).annotate({ discriminator: "type" }),
  ),
})
export type PromptInput = Schema.Schema.Type<typeof PromptInput>

export class LoopInput extends Schema.Class<LoopInput>("SessionPrompt.LoopInput")({
  sessionID: SessionID,
}) {}

export const ShellInput = Schema.Struct({
  sessionID: SessionID,
  messageID: Schema.optional(MessageID),
  agent: Schema.String,
  model: Schema.optional(ModelRef),
  command: Schema.String,
})
export type ShellInput = Schema.Schema.Type<typeof ShellInput>

export const CommandInput = Schema.Struct({
  messageID: Schema.optional(MessageID),
  sessionID: SessionID,
  agent: Schema.optional(Schema.String),
  model: Schema.optional(Schema.String),
  arguments: Schema.String,
  command: Schema.String,
  variant: Schema.optional(Schema.String),
  format: Schema.optional(SessionV1.Format),
  // Inlined (no identifier annotation) to keep the original SDK output — the
  // PromptInput call site below references FilePartInput by ref via the
  // Schema export in message-v2.ts.
  parts: Schema.optional(
    Schema.Array(
      Schema.Union([
        Schema.Struct({
          id: Schema.optional(PartID),
          type: Schema.Literal("file"),
          mime: Schema.String,
          filename: Schema.optional(Schema.String),
          url: Schema.String,
          source: Schema.optional(SessionV1.FilePartSource),
        }),
      ]).annotate({ discriminator: "type" }),
    ),
  ),
})
export type CommandInput = Schema.Schema.Type<typeof CommandInput>

/**
 * ARPOS: the structured-output tool is a PASS-THROUGH.
 *
 * Validation and the corrective-retry budget live in the ARPOS orchestrator,
 * the single authority for structured output across both of its engines — so an
 * `output_schema` behaves identically on OpenCode and on Claude Code (which
 * reaches the same endpoint through an MCP tool). What stays here is the one
 * thing only the fork can do: hand the model the caller's REAL JSON Schema as
 * the tool's input schema, via the pass-through `jsonSchema()`.
 *
 * `submit` returns the text the orchestrator wants the model to read — the
 * success line, a corrective error list, or a terminal refusal — whether the
 * answer was accepted, and whether the turn is over either way (`onTurnOver`).
 *
 * @internal Exported for testing
 */
export function createStructuredOutputTool(input: {
  schema: Record<string, any>
  submit: (args: unknown) => Promise<StructuredSubmitResult>
  onTurnOver: () => void
}): AITool {
  // Remove $schema property if present (not needed for tool input)
  const { $schema: _, ...toolSchema } = input.schema

  return tool({
    description: STRUCTURED_OUTPUT_DESCRIPTION,
    inputSchema: jsonSchema(toolSchema as JSONSchema7),
    async execute(args) {
      const verdict = await input.submit(args)
      if (verdict.accepted || verdict.terminal) input.onTurnOver()
      return {
        output: verdict.text,
        title: verdict.accepted ? "Structured Output" : "Structured Output (rejected)",
        metadata: { valid: verdict.accepted },
      }
    },
    toModelOutput({ output }) {
      return {
        type: "text",
        value: output.output,
      }
    },
  })
}
const bashRegex = /!`([^`]+)`/g
// Match [Image N] as single token, quoted strings, or non-space sequences
const argsRegex = /(?:\[Image\s+\d+\]|"[^"]*"|'[^']*'|[^\s"']+)/gi
const placeholderRegex = /\$(\d+)/g
const quoteTrimRegex = /^["']|["']$/g
const slashRegex = /^\/(\S+)(?:\s+([\s\S]*))?$/

export const node = LayerNode.make({
  service: Service,
  layer: layer,
  deps: [
    SessionStatus.node,
    Session.node,
    Agent.node,
    Provider.node,
    SessionProcessor.node,
    SessionCompaction.node,
    Plugin.node,
    Command.node,
    Config.node,
    Permission.node,
    FSUtil.node,
    MCP.node,
    LSP.node,
    ToolRegistry.node,
    Truncate.node,
    Image.node,
    CrossSpawnSpawner.node,
    Instruction.node,
    SessionRunState.node,
    SessionRevert.node,
    SessionSummary.node,
    SystemPrompt.node,
    LLM.node,
    EventV2Bridge.node,
    RuntimeFlags.node,
    Database.node,
  ],
})

export * as SessionPrompt from "./prompt"
