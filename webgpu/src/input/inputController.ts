import type { DSLCommand } from '../dsl/commands'
import type { WorldEngine } from '../engine/worldEngine'

export class InputController {
  constructor(private engine: WorldEngine) {}

  dispatch(command: DSLCommand) {
    this.engine.apply(command)
  }

  dispose() {
    // Placeholder for future resource cleanup.
  }
}

export function createInputController(engine: WorldEngine) {
  return new InputController(engine)
}
