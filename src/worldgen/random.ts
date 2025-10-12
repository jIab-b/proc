export type RandomStats = {
  count: number
  min: number
  max: number
  mean: number
}

export class StageRandom {
  private state: number
  private stats: RandomStats

  constructor(public readonly seed: number) {
    this.state = seed >>> 0
    this.stats = { count: 0, min: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY, mean: 0 }
  }

  next() {
    let t = this.state += 0x6d2b79f5
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    const result = ((t ^ (t >>> 14)) >>> 0) / 4294967296
    this.accumulate(result)
    return result
  }

  nextRange(min: number, max: number) {
    return min + (max - min) * this.next()
  }

  nextInt(maxExclusive: number) {
    return Math.floor(this.next() * maxExclusive)
  }

  private accumulate(value: number) {
    const { stats } = this
    stats.count += 1
    stats.min = Math.min(stats.min, value)
    stats.max = Math.max(stats.max, value)
    const delta = value - stats.mean
    stats.mean += delta / stats.count
  }

  snapshot(): RandomStats {
    if (this.stats.count === 0) {
      return { count: 0, min: 0, max: 0, mean: 0 }
    }
    return { ...this.stats }
  }
}
