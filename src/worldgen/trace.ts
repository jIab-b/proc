export interface TraceEntry {
  stage: string
  seed: number
  startedAt: number
  durationMs?: number
  data: Record<string, unknown>
}

export class TraceScope {
  private closed = false

  constructor(private readonly ctx: TracingContext, private readonly entry: TraceEntry) {}

  add(data: Record<string, unknown>) {
    Object.assign(this.entry.data, data)
  }

  end(extra: Record<string, unknown> = {}) {
    if (this.closed) return
    this.closed = true
    this.add(extra)
    this.entry.durationMs = performance.now() - this.entry.startedAt
    this.ctx.commit(this.entry)
  }
}

export class TracingContext {
  private stages: TraceEntry[] = []

  constructor(public readonly label: string) {}

  begin(stage: string, seed: number, initial: Record<string, unknown> = {}) {
    const entry: TraceEntry = {
      stage,
      seed,
      startedAt: performance.now(),
      data: { ...initial }
    }
    return new TraceScope(this, entry)
  }

  commit(entry: TraceEntry) {
    this.stages.push(entry)
  }

  serialize() {
    return {
      label: this.label,
      stages: this.stages.map((stage) => ({
        stage: stage.stage,
        seed: stage.seed,
        durationMs: stage.durationMs,
        data: stage.data
      }))
    }
  }

  get entries() {
    return this.stages
  }
}

export function createTracingContext(label: string) {
  return new TracingContext(label)
}
