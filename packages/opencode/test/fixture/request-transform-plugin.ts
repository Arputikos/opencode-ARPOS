// ARPOS (patch 29): masks a word in every string of each request's messages, the way
// ARPOS's plugin masks known secrets. Every export must be a plugin function.
export default async () => ({
  "experimental.chat.request.transform": async (_input: unknown, output: { messages: unknown[] }) => {
    const mask = (node: unknown): unknown =>
      typeof node === "string"
        ? node.replaceAll("s3cr3t-value", "[REDACTED]")
        : Array.isArray(node)
          ? node.map(mask)
          : node && typeof node === "object"
            ? Object.fromEntries(Object.entries(node).map(([k, v]) => [k, mask(v)]))
            : node
    output.messages.splice(0, output.messages.length, ...(mask(output.messages) as unknown[]))
  },
})
