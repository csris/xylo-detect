import { describe, test, expect } from 'vitest'
import { infernoColor } from './utils'

// ── infernoColor ──────────────────────────────────────────────────────────────

describe('infernoColor', () => {
  test('t=0 → [0, 0, 4]', () => {
    expect(infernoColor(0)).toEqual([0, 0, 4])
  })

  test('t=1 → [252, 255, 164]', () => {
    expect(infernoColor(1)).toEqual([252, 255, 164])
  })

  test('t<0 clamps to [0, 0, 4]', () => {
    expect(infernoColor(-0.001)).toEqual([0, 0, 4])
    expect(infernoColor(-100)).toEqual([0, 0, 4])
  })

  test('t>1 clamps to [252, 255, 164]', () => {
    expect(infernoColor(1.001)).toEqual([252, 255, 164])
    expect(infernoColor(100)).toEqual([252, 255, 164])
  })

  test('t=0.25 → exact stop [58, 9, 99]', () => {
    expect(infernoColor(0.25)).toEqual([58, 9, 99])
  })

  test('t=0.50 → exact stop [188, 55, 84]', () => {
    expect(infernoColor(0.5)).toEqual([188, 55, 84])
  })

  test('t=0.75 → exact stop [252, 137, 97]', () => {
    expect(infernoColor(0.75)).toEqual([252, 137, 97])
  })

  test('t=0.125 → midpoint of segment 0→1', () => {
    // u=0.5: [round(29), round(4.5)→5, round(51.5)→52]
    expect(infernoColor(0.125)).toEqual([29, 5, 52])
  })

  test('t=0.375 → midpoint of segment 1→2', () => {
    // u=0.5: [round(123), round(32), round(91.5)→92]
    expect(infernoColor(0.375)).toEqual([123, 32, 92])
  })

  test('t=0.625 → midpoint of segment 2→3', () => {
    // u=0.5: [round(220), round(96), round(90.5)→91]
    expect(infernoColor(0.625)).toEqual([220, 96, 91])
  })

  test('t=0.875 → midpoint of segment 3→4', () => {
    // u=0.5: [round(252), round(196), round(130.5)→131]
    expect(infernoColor(0.875)).toEqual([252, 196, 131])
  })

  test('returns a tuple of exactly 3 elements', () => {
    const result = infernoColor(0.5)
    expect(result).toHaveLength(3)
  })

  test('all components are integers', () => {
    for (const t of [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]) {
      const [r, g, b] = infernoColor(t)
      expect(Number.isInteger(r)).toBe(true)
      expect(Number.isInteger(g)).toBe(true)
      expect(Number.isInteger(b)).toBe(true)
    }
  })

  test('all components in [0, 255]', () => {
    for (let i = 0; i <= 20; i++) {
      const [r, g, b] = infernoColor(i / 20)
      expect(r).toBeGreaterThanOrEqual(0)
      expect(r).toBeLessThanOrEqual(255)
      expect(g).toBeGreaterThanOrEqual(0)
      expect(g).toBeLessThanOrEqual(255)
      expect(b).toBeGreaterThanOrEqual(0)
      expect(b).toBeLessThanOrEqual(255)
    }
  })

  test('brightness increases from t=0 to t=0.5 to t=1', () => {
    const luma = (t: number) => { const [r, g, b] = infernoColor(t); return r + g + b }
    expect(luma(1)).toBeGreaterThan(luma(0.5))
    expect(luma(0.5)).toBeGreaterThan(luma(0))
  })

  test('t=0.001 is close to the dark end', () => {
    const [r, g, b] = infernoColor(0.001)
    expect(r).toBeLessThan(5)
    expect(g).toBeLessThan(5)
    expect(b).toBeGreaterThanOrEqual(4)
  })

  test('t=0.999 is close to the bright end', () => {
    const [r] = infernoColor(0.999)
    expect(r).toBeGreaterThan(248)
  })
})
