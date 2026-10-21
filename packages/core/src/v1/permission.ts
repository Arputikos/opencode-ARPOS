export * as PermissionV1 from "./permission"

import { Schema } from "effect"
export * from "@opencode-ai/schema/permission-v1"
import { ID } from "@opencode-ai/schema/permission-v1"

export class RejectedError extends Schema.TaggedErrorClass<RejectedError>()("PermissionRejectedError", {}) {
  override get message() {
    return "The user rejected permission to use this specific tool call."
  }
}

export class CorrectedError extends Schema.TaggedErrorClass<CorrectedError>()("PermissionCorrectedError", {
  feedback: Schema.String,
}) {
  // ARPOS: the feedback IS the message, verbatim. The orchestrator is the only
  // one replying with a message, and it words every refusal itself (operator
  // refusal, always-deny rule, gate failure) — the same text the Claude Code
  // engine hands the model. The upstream frame ("The user rejected … with the
  // following feedback: …") attributed a static rule to a user.
  override get message() {
    return this.feedback
  }
}

export class DeniedError extends Schema.TaggedErrorClass<DeniedError>()("PermissionDeniedError", {
  ruleset: Schema.Any,
}) {
  override get message() {
    return `The user has specified a rule which prevents you from using this specific tool call. Here are some of the relevant rules ${JSON.stringify(this.ruleset)}`
  }
}

export class NotFoundError extends Schema.TaggedErrorClass<NotFoundError>()("Permission.NotFoundError", {
  requestID: ID,
}) {}

export type Error = DeniedError | RejectedError | CorrectedError
