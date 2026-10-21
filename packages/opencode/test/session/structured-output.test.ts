import { describe, expect, test } from "bun:test"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Exit, Schema } from "effect"
import { MessageV2 } from "../../src/session/message-v2"
import { SessionPrompt } from "../../src/session/prompt"
import { SessionID, MessageID } from "../../src/session/schema"

const decodeFormat = Schema.decodeUnknownExit(SessionV1.Format)
const decodeUser = Schema.decodeUnknownExit(SessionV1.User)
const decodeAssistant = Schema.decodeUnknownExit(SessionV1.Assistant)

describe("structured-output.OutputFormat", () => {
  test("encodes a format read back from storage as a plain object", () => {
    // ARPOS: a user message's `format` comes from SQLite as plain JSON; the
    // message list encodes it — a Schema.Class here made that a 400.
    const encode = Schema.encodeUnknownExit(SessionV1.User)
    const stored = {
      id: MessageID.make("msg_123"),
      sessionID: SessionID.make("ses_123"),
      role: "user",
      time: { created: Date.now() },
      agent: "default",
      model: { providerID: "anthropic", modelID: "claude-3" },
      format: { type: "json_schema", schema: { type: "object" }, retryCount: 3 },
    }
    expect(Exit.isSuccess(encode(stored))).toBe(true)
  })

  test("parses text format", () => {
    const result = decodeFormat({ type: "text" })
    expect(Exit.isSuccess(result)).toBe(true)
    if (Exit.isSuccess(result)) {
      expect(result.value.type).toBe("text")
    }
  })

  test("parses json_schema format with defaults", () => {
    const result = decodeFormat({
      type: "json_schema",
      schema: { type: "object", properties: { name: { type: "string" } } },
    })
    expect(Exit.isSuccess(result)).toBe(true)
    if (Exit.isSuccess(result)) {
      expect(result.value.type).toBe("json_schema")
      if (result.value.type === "json_schema") {
        expect(result.value.retryCount).toBe(2) // default value
      }
    }
  })

  test("parses json_schema format with custom retryCount", () => {
    const result = decodeFormat({
      type: "json_schema",
      schema: { type: "object" },
      retryCount: 5,
    })
    expect(Exit.isSuccess(result)).toBe(true)
    if (Exit.isSuccess(result) && result.value.type === "json_schema") {
      expect(result.value.retryCount).toBe(5)
    }
  })

  test("rejects invalid type", () => {
    const result = decodeFormat({ type: "invalid" })
    expect(Exit.isFailure(result)).toBe(true)
  })

  test("rejects json_schema without schema", () => {
    const result = decodeFormat({ type: "json_schema" })
    expect(Exit.isFailure(result)).toBe(true)
  })

  test("rejects negative retryCount", () => {
    const result = decodeFormat({
      type: "json_schema",
      schema: { type: "object" },
      retryCount: -1,
    })
    expect(Exit.isFailure(result)).toBe(true)
  })
})

describe("structured-output.StructuredOutputError", () => {
  test("creates error with message and retries", () => {
    const error = new SessionV1.StructuredOutputError({
      message: "Failed to validate",
      retries: 3,
    })

    expect(error.name).toBe("StructuredOutputError")
    expect(error.data.message).toBe("Failed to validate")
    expect(error.data.retries).toBe(3)
  })

  test("converts to object correctly", () => {
    const error = new SessionV1.StructuredOutputError({
      message: "Test error",
      retries: 2,
    })

    const obj = error.toObject()
    expect(obj.name).toBe("StructuredOutputError")
    expect(obj.data.message).toBe("Test error")
    expect(obj.data.retries).toBe(2)
  })

  test("isInstance correctly identifies error", () => {
    const error = new SessionV1.StructuredOutputError({
      message: "Test",
      retries: 1,
    })

    expect(SessionV1.StructuredOutputError.isInstance(error)).toBe(true)
    expect(SessionV1.StructuredOutputError.isInstance({ name: "other" })).toBe(false)
  })
})

describe("structured-output.UserMessage", () => {
  test("user message accepts outputFormat", () => {
    const result = decodeUser({
      id: MessageID.ascending(),
      sessionID: SessionID.descending(),
      role: "user",
      time: { created: Date.now() },
      agent: "default",
      model: { providerID: "anthropic", modelID: "claude-3" },
      outputFormat: {
        type: "json_schema",
        schema: { type: "object" },
      },
    })
    expect(Exit.isSuccess(result)).toBe(true)
  })

  test("user message works without outputFormat (optional)", () => {
    const result = decodeUser({
      id: MessageID.ascending(),
      sessionID: SessionID.descending(),
      role: "user",
      time: { created: Date.now() },
      agent: "default",
      model: { providerID: "anthropic", modelID: "claude-3" },
    })
    expect(Exit.isSuccess(result)).toBe(true)
  })
})

describe("structured-output.AssistantMessage", () => {
  const baseAssistantMessage = {
    id: MessageID.ascending(),
    sessionID: SessionID.descending(),
    role: "assistant" as const,
    parentID: MessageID.ascending(),
    modelID: "claude-3",
    providerID: "anthropic",
    mode: "default",
    agent: "default",
    path: { cwd: "/test", root: "/test" },
    cost: 0.001,
    tokens: { input: 100, output: 50, reasoning: 0, cache: { read: 0, write: 0 } },
    time: { created: Date.now() },
  }

  test("assistant message accepts structured", () => {
    const result = decodeAssistant({
      ...baseAssistantMessage,
      structured: { company: "Anthropic", founded: 2021 },
    })
    expect(Exit.isSuccess(result)).toBe(true)
    if (Exit.isSuccess(result)) {
      expect(result.value.structured).toEqual({ company: "Anthropic", founded: 2021 })
    }
  })

  test("assistant message works without structured_output (optional)", () => {
    const result = decodeAssistant(baseAssistantMessage)
    expect(Exit.isSuccess(result)).toBe(true)
  })
})

describe("structured-output.createStructuredOutputTool", () => {
  // Patched (arpos): the tool is a PASS-THROUGH. Validation, the corrective
  // retry budget and the failure all live in the ARPOS orchestrator, so the
  // fork is only responsible for two things — showing the model the caller's
  // real JSON Schema, and relaying the orchestrator's verdict back. The
  // validation behaviour these tests used to assert is covered by the
  // orchestrator's own suite (`core/__tests__/structured-output.test.ts`).
  const accept = async () => ({ accepted: true, terminal: false, text: "Structured output captured successfully." })
  const reject = async () => ({ accepted: false, terminal: false, text: "did NOT match the required schema" })
  const callOpts = { toolCallId: "test-call-id", messages: [], abortSignal: undefined as any }

  test("creates tool with description", () => {
    const tool = SessionPrompt.createStructuredOutputTool({
      schema: { type: "object" },
      submit: accept,
      onTurnOver: () => {},
    })

    expect(tool.description).toContain("structured format")
  })

  test("hands the model the caller's schema verbatim as inputSchema", () => {
    const schema = {
      type: "object",
      properties: {
        company: { type: "string" },
        founded: { type: "number" },
        tags: { type: "array", items: { type: "string" } },
        user: { type: "object", properties: { name: { type: "string" } }, required: ["name"] },
      },
      required: ["company"],
    }

    const tool = SessionPrompt.createStructuredOutputTool({
      schema,
      submit: accept,
      onTurnOver: () => {},
    })

    // This is the one thing only the fork can do: the pass-through `jsonSchema()`
    // bypasses the zod-only tool registry, so oneOf/$ref/nested shapes survive.
    const inputSchema = tool.inputSchema as any
    expect(inputSchema.jsonSchema?.properties?.company).toBeDefined()
    expect(inputSchema.jsonSchema?.properties?.founded?.type).toBe("number")
    expect(inputSchema.jsonSchema?.properties?.tags?.items?.type).toBe("string")
    expect(inputSchema.jsonSchema?.properties?.user?.required).toContain("name")
    expect(inputSchema.jsonSchema?.required).toContain("company")
  })

  test("strips $schema property from inputSchema", () => {
    const tool = SessionPrompt.createStructuredOutputTool({
      schema: {
        $schema: "http://json-schema.org/draft-07/schema#",
        type: "object",
        properties: { name: { type: "string" } },
      },
      submit: accept,
      onTurnOver: () => {},
    })

    const inputSchema = tool.inputSchema as any
    expect(inputSchema.jsonSchema?.$schema).toBeUndefined()
  })

  test("execute submits the model's argument verbatim and reports acceptance", async () => {
    let submitted: unknown
    let accepted = false

    const tool = SessionPrompt.createStructuredOutputTool({
      schema: { type: "object", properties: { name: { type: "string" } } },
      submit: async (args) => {
        submitted = args
        return { accepted: true, terminal: false, text: "Structured output captured successfully." }
      },
      onTurnOver: () => {
        accepted = true
      },
    })

    const testArgs = { name: "Test Company", nested: { deep: [1, 2, 3] } }
    const result = await tool.execute!(testArgs, callOpts)

    expect(submitted).toEqual(testArgs)
    expect(accepted).toBe(true)
    expect(result.output).toBe("Structured output captured successfully.")
    expect(result.metadata.valid).toBe(true)
  })

  test("a rejected submission returns the orchestrator's text and does NOT end the turn", async () => {
    let accepted = false

    const tool = SessionPrompt.createStructuredOutputTool({
      schema: { type: "object", required: ["sentiment"] },
      submit: reject,
      onTurnOver: () => {
        accepted = true
      },
    })

    const result = await tool.execute!({ sentiment: "banana" }, callOpts)

    // `onTurnOver` is what breaks the run loop — a rejection with budget left
    // must leave the model free to correct itself on the next step.
    expect(accepted).toBe(false)
    expect(result.metadata.valid).toBe(false)
    expect(result.output).toContain("did NOT match the required schema")
  })

  test("a TERMINAL refusal ends the turn — under toolChoice: required the model could not stop on its own", async () => {
    let over = false

    const tool = SessionPrompt.createStructuredOutputTool({
      schema: { type: "object", required: ["sentiment"] },
      submit: async () => ({ accepted: false, terminal: true, text: "no corrective attempts left" }),
      onTurnOver: () => {
        over = true
      },
    })

    const result = await tool.execute!({ sentiment: "banana" }, callOpts)

    expect(over).toBe(true)
    // Still a refusal: the orchestrator's gate fails the turn, nothing here accepts it.
    expect(result.metadata.valid).toBe(false)
    expect(result.output).toContain("no corrective attempts left")
  })

  test("toModelOutput returns text value", async () => {
    const tool = SessionPrompt.createStructuredOutputTool({
      schema: { type: "object" },
      submit: accept,
      onTurnOver: () => {},
    })

    expect(tool.toModelOutput).toBeDefined()
    const modelOutput = await Promise.resolve(
      tool.toModelOutput!({
        toolCallId: "test-call-id",
        input: {},
        output: {
          output: "Test output",
        },
      }),
    )

    expect(modelOutput.type).toBe("text")
    if (modelOutput.type !== "text") throw new Error("expected text model output")
    expect(modelOutput.value).toBe("Test output")
  })
})
