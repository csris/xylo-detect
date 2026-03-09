export function infernoColor(t: number): readonly [number, number, number] {
  const stops = [
    [0.00,   0,   0,   4],
    [0.25,  58,   9,  99],
    [0.50, 188,  55,  84],
    [0.75, 252, 137,  97],
    [1.00, 252, 255, 164],
  ] as const
  if (t <= 0) return [0, 0, 4]
  if (t >= 1) return [252, 255, 164]
  let i = 1
  while (i < stops.length - 1 && (stops[i]?.[0] ?? 0) < t) i++
  const s0 = stops[i - 1]
  const s1 = stops[i]
  if (!s0 || !s1) return [0, 0, 0]
  const u = (t - s0[0]) / (s1[0] - s0[0])
  return [
    Math.round(s0[1] + u * (s1[1] - s0[1])),
    Math.round(s0[2] + u * (s1[2] - s0[2])),
    Math.round(s0[3] + u * (s1[3] - s0[3])),
  ]
}
